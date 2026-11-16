"""Exact data parity with the cached, checksum-pinned upstream loader."""

import ast
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from exp.datasets.hypergraph import HYPERGRAPHS
from exp.hsnn_run import load_data, load_provenance, load_splits

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "name",
    [n for n, s in HYPERGRAPHS.items() if n != "cora-hypergraph"],
)
def test_upstream_data_parity(name):
    source = Path("tmp/hsnn-cora-source/data_utils_direction.py")
    artifact = Path("data/hypergraphs") / f"{name}.npz"
    if not source.exists() or not artifact.exists():
        pytest.skip("Reference source or prepared data unavailable")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "5d8fe365db1450ce72291c8b5f722c1099777b1e2c4e452baef7847f2bff4508"
    )
    spec = HYPERGRAPHS[name]
    scope = dict(torch=torch, np=np, pickle=pickle, os=os, Data=Data, tqdm=lambda x: x)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope
    )
    family = spec.source_directory.split("/")[0]
    loaders = {
        "additional": "load_additional_dataset",
        "cocitation": "load_citation_dataset_direction",
        "mail": "load_mail_dataset_direction",
        "protein": "load_directed_molecular_dataset",
    }
    if family == "other":
        from torch_geometric_signed_directed.data import load_directed_real_data

        graph = load_directed_real_data(
            dataset="telegram" if spec.reference_name == "telegram" else "webkb",
            root="data/hypergraphs/raw/other",
            name=spec.reference_name,
        )
        official = scope["load_other_directed_graph"](graph)
    else:
        official = scope[loaders[family]](
            str(Path("data/hypergraphs/raw") / family), spec.reference_name
        )
    mask = official.edge_index[0] < official.num_nodes
    index = official.edge_index[:, mask].clone()
    # processing2 (additional datasets) already returns V->E zero-based
    # incidences; the other loaders return a doubled bipartite graph.
    if family not in ("additional", "other"):
        index[1] -= official.num_nodes
    order = (index[0] * official.num_hyperedges + index[1]).argsort()
    x, y, actual, edges = load_data(artifact)
    torch.testing.assert_close(actual, index[:, order])
    assert edges == official.num_hyperedges
    torch.testing.assert_close(y, official.y)
    if official.x is not None:
        torch.testing.assert_close(x, official.x, rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            x[:, 0], torch.bincount(actual[0], minlength=x.size(0)).float()
        )
    with np.load(artifact) as a:
        np.testing.assert_array_equal(
            a["incidence_roles"], official.edge_weight[mask][order].numpy()
        )
    provenance = load_provenance(artifact)
    splits = load_splits(
        artifact.with_suffix(".splits.npz"), len(y), 10, provenance["dataset_sha256"]
    )
    if spec.split == "predefined":
        for fold, parts in enumerate(splits):
            for key, actual_part in zip(("train", "val", "test"), parts, strict=True):
                expected = getattr(official, f"{key}_mask")[:, fold].nonzero().flatten()
                torch.testing.assert_close(actual_part, expected)
    else:
        utils = Path("tmp/hsnn-cora-source/utils.py")
        assert (
            hashlib.sha256(utils.read_bytes()).hexdigest()
            == "a0401ae4bf142301b1fdf612ce63c7379783bdedd07c0d7d73297c4b40a688db"
        )
        definition = [
            n
            for n in ast.parse(utils.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name == "rand_train_test_idx_2"
        ]
        exec(
            compile(ast.Module(body=definition, type_ignores=[]), str(utils), "exec"),
            scope,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for parts in splits:
                expected = scope["rand_train_test_idx_2"](
                    official.y, 0.5, 0.25, balance=False
                )
                for key, actual_part in zip(
                    ("train", "valid", "test"), parts, strict=True
                ):
                    torch.testing.assert_close(actual_part, expected[key])
