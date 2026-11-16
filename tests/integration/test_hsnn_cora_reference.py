"""Optional exact parity with pinned, locally cached official Cora loader code.

Set HSNN_CORA_REFERENCE_DIR to a cache containing the official loader, utils,
cora.content, cora.cites, features.pickle and labels.pickle. No network is used.
Only the four relevant upstream function definitions are executed, unchanged;
unrelated upstream model imports and environment dependencies are avoided.
"""

import ast
import hashlib
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.prepare_hsnn_cora import RAW_SHA256, build_incidence, make_splits, prepare
from torch_geometric.data import Data

pytestmark = pytest.mark.integration


def extract_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_pinned_official_loader_and_split_parity(tmp_path):
    cache = Path(os.environ.get("HSNN_CORA_REFERENCE_DIR", "tmp/hsnn-cora-source"))
    expected = {
        **RAW_SHA256,
        "features.pickle": (
            "a20ab3c4903d6cf97d277f5ce8d26735e31d08c888cdd11e3f0a5bb78d7c0602"
        ),
        "labels.pickle": (
            "0e6edd0d971dde22c1116fa662362262163a7a6d557eafd60ac6857de4569527"
        ),
        "data_utils_direction.py": (
            "5d8fe365db1450ce72291c8b5f722c1099777b1e2c4e452baef7847f2bff4508"
        ),
        "utils.py": (
            "a0401ae4bf142301b1fdf612ce63c7379783bdedd07c0d7d73297c4b40a688db"
        ),
    }
    if any(not (cache / name).exists() for name in expected):
        pytest.skip("pinned upstream Cora cache not available")
    for name, digest in expected.items():
        assert hashlib.sha256((cache / name).read_bytes()).hexdigest() == digest, name
    raw = tmp_path / "cora"
    raw.mkdir()
    for name in ("cora.content", "cora.cites", "features.pickle", "labels.pickle"):
        shutil.copyfile(cache / name, raw / name)
    scope = extract_functions(
        cache / "data_utils_direction.py",
        [
            "count_duplicates_ignore_weight",
            "deduplicate_ignore_weight",
            "processing",
            "load_citation_dataset_direction",
        ],
        dict(torch=torch, np=np, pickle=pickle, os=os, Data=Data, tqdm=lambda x: x),
    )
    official = scope["load_citation_dataset_direction"](str(tmp_path), "cora")
    mask = official.edge_index[0] < official.num_nodes
    index = official.edge_index[:, mask].clone()
    index[1] -= official.num_nodes
    order = (index[0] * official.num_hyperedges + index[1]).argsort()
    content = np.genfromtxt(raw / "cora.content", dtype=str)
    actual, roles, _ = build_incidence(
        content[:, 0].tolist(), (raw / "cora.cites").read_text().splitlines()
    )
    np.testing.assert_array_equal(actual, index[:, order].numpy())
    np.testing.assert_array_equal(roles, official.edge_weight[mask][order].numpy())
    artifact = prepare(raw, tmp_path / "prepared")
    with np.load(artifact, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"], official.x.numpy())
        np.testing.assert_array_equal(data["y"], official.y.numpy())
        assert data["num_hyperedges"].item() == official.num_hyperedges == 1565
    split_fn = extract_functions(
        cache / "utils.py", ["rand_train_test_idx_2"], dict(torch=torch)
    )["rand_train_test_idx_2"]
    actual_splits = make_splits(2708, 10, 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for row in range(10):
            official_split = split_fn(official.y, 0.5, 0.25, balance=False)
            for key, source in (("train", "train"), ("val", "valid"), ("test", "test")):
                np.testing.assert_array_equal(
                    actual_splits[key][row], official_split[source].numpy()
                )
