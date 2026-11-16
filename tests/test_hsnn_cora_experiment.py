"""Cora topology, held-out splits and checkpoint-selection protocol checks."""

import argparse
import hashlib
import json

import numpy as np
import pytest
import torch
from scripts.prepare_hsnn_cora import build_incidence, make_splits, prepare

from exp.hsnn_run import load_data, load_provenance, load_splits, train_run


def test_citation_topology_includes_tail_and_preserves_edge_identity():
    index, roles, tails = build_incidence(
        ["a", "b", "c", "d"],
        ["b a", "b c", "b a", "a b", "c c", "unknown a"],
    )
    # First-seen tails: b, a, c. A self citation is excluded from the heads,
    # but its tail edge remains, matching the reference's insertion order.
    np.testing.assert_array_equal(tails, [1, 0, 2])
    np.testing.assert_array_equal(index, [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 2]])
    np.testing.assert_array_equal(roles, [1, -1, -1, 1, 1, -1])


def test_split_bank_replays_global_reference_stream_without_mutating_rng():
    initial = torch.get_rng_state().clone()
    actual = make_splits(20, 3, 0)
    assert torch.equal(initial, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        expected = [torch.randperm(20).numpy() for _ in range(3)]
    for row, permutation in enumerate(expected):
        joined = np.concatenate([actual[k][row] for k in ("train", "val", "test")])
        np.testing.assert_array_equal(joined, permutation)


@pytest.mark.parametrize("problem", ["overlap", "range", "float", "dataset", "runs"])
def test_bad_split_bank_rejected(tmp_path, problem):
    arrays = make_splits(20, 2, 0)
    digest = "data-digest"
    if problem == "overlap":
        arrays["test"][0, 0] = arrays["train"][0, 0]
    elif problem == "range":
        arrays["test"][0, 0] = 20
    elif problem == "float":
        arrays["train"] = arrays["train"].astype(np.float32)
    elif problem == "dataset":
        digest = "wrong-data"
    path = tmp_path / "splits.npz"
    np.savez(
        path,
        train=arrays["train"],
        val=arrays["val"],
        test=arrays["test"],
        dataset_sha256=np.array(digest),
    )
    with pytest.raises(ValueError):
        load_splits(path, 20, 3 if problem == "runs" else 2, "data-digest")


def test_old_pickle_derived_artifact_is_not_accepted(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, x=np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="legacy Cora"):
        load_data(path)


def test_raw_checksum_failure_prevents_preparation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "cora.content").write_text("incorrect source")
    with pytest.raises(ValueError, match="checksum"):
        prepare(raw, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_provenance_detects_artifact_changes(tmp_path):
    path = tmp_path / "data.npz"
    splits = path.with_suffix(".splits.npz")
    path.write_bytes(b"data")
    splits.write_bytes(b"splits")
    metadata = {
        "dataset_id": "dshn-cora-q0-v1",
        "reference_commit": "d94f81bc7d86bd8d41e416083063f3446c1ad796",
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256(splits.read_bytes()).hexdigest(),
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(metadata))
    assert load_provenance(path) == metadata
    splits.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_provenance(path)


@pytest.mark.parametrize("evaluate_test", [False, True])
def test_ties_keep_first_checkpoint_and_test_is_opt_in(
    tmp_path, monkeypatch, evaluate_test
):
    args = argparse.Namespace(
        output=tmp_path,
        stalk_dim=2,
        hidden_dim=3,
        num_layers=1,
        variant="diagonal",
        dropout=0.0,
        residual=False,
        degree_regularization=0.1,
        lr=0.01,
        weight_decay=0.0,
        epochs=5,
        patience=1,
        evaluate_test=evaluate_test,
    )
    x = torch.randn(8, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
    index = torch.stack((torch.arange(8), torch.zeros(8, dtype=torch.long)))
    train, val, test = torch.arange(4), torch.tensor([4, 5]), torch.tensor([6, 7])
    calls = []

    def fake_metrics(logits, labels):
        calls.append(labels.clone())
        return {"accuracy": 0.5, "loss": 1.0 / len(calls)}

    monkeypatch.setattr("exp.hsnn_run.metrics", fake_metrics)
    result = train_run(args, (x, y, index, 1), 0, (train, val, test), 0)
    assert result["best_epoch"] == 1  # Lower loss at equal accuracy cannot win.
    assert result["epochs_ran"] == 2  # A tie counts toward patience.
    assert len(calls) == 2 + int(evaluate_test)
    assert all(torch.equal(labels, y[val]) for labels in calls[:2])
    if evaluate_test:
        assert torch.equal(calls[-1], y[test])
    assert (result["test"] is not None) == evaluate_test
    saved = torch.load(tmp_path / "run-00-seed-0/best.pt", weights_only=True)
    assert saved["epoch"] == 1
