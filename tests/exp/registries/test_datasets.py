# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/registries/datasets.py."""

import dataclasses

import pytest

from exp.registries.datasets import DatasetEntry, DatasetRegistry, dataset_registry

# All datasets expected in the registry: the 16 undirected benchmarks plus the
# directed-only sources (telegram and the 24 DSBM synthetic graphs).
_SYNTHETIC = {
    f"dataset_nodes500_alpha{a}_beta{b}"
    for a in ("0.05", "0.08", "0.1")
    for b in ("0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4")
}
_ALL = {
    "cora",
    "citeseer",
    "pubmed",
    "chameleon",
    "squirrel",
    "chameleon_filtered",
    "squirrel_filtered",
    "cornell",
    "texas",
    "wisconsin",
    "film",
    "amazon_ratings",
    "minesweeper",
    "questions",
    "roman_empire",
    "tolokers",
    "telegram",
    "ogbn_arxiv",
    "ogbn_products",
    "ogbn_proteins",
    "pokec",
} | _SYNTHETIC

_NPZ = {
    "cora",
    "citeseer",
    "pubmed",
    "chameleon",
    "squirrel",
    "cornell",
    "texas",
    "wisconsin",
    "film",
}
_ROC_AUC = {"minesweeper", "questions", "tolokers", "ogbn_proteins"}

# Datasets that do not carry the default 10 folds: OGB ships one official
# split; LINKX pokec ships five.
_NUM_SPLITS_OVERRIDES = {
    "ogbn_arxiv": 1,
    "ogbn_products": 1,
    "ogbn_proteins": 1,
    "pokec": 5,
}


class TestDatasetRegistryContents:
    def test_all_datasets_registered(self):
        assert set(dataset_registry.list_keys()) == _ALL

    def test_npz_split_datasets(self):
        npz = {
            name
            for name in dataset_registry.list_keys()
            if dataset_registry.get(name).split_type == "npz_file"
        }
        assert npz == _NPZ

    def test_pyg_mask_datasets(self):
        pyg = {
            name
            for name in dataset_registry.list_keys()
            if dataset_registry.get(name).split_type == "pyg_mask"
        }
        assert pyg == _ALL - _NPZ

    def test_roc_auc_datasets(self):
        roc = {
            name
            for name in dataset_registry.list_keys()
            if dataset_registry.get(name).metric == "roc_auc"
        }
        assert roc == _ROC_AUC

    def test_acc_datasets(self):
        acc = {
            name
            for name in dataset_registry.list_keys()
            if dataset_registry.get(name).metric == "acc"
        }
        assert acc == _ALL - _ROC_AUC

    def test_num_splits(self):
        for name in dataset_registry.list_keys():
            expected = _NUM_SPLITS_OVERRIDES.get(name, 10)
            assert dataset_registry.get(name).num_splits == expected

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            dataset_registry.get("unknown_dataset")


class TestDatasetEntry:
    def test_frozen(self):
        entry = DatasetEntry(metric="acc", split_type="npz_file")
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.metric = "roc_auc"  # ty: ignore[invalid-assignment]

    def test_num_splits_default(self):
        entry = DatasetEntry(metric="acc", split_type="npz_file")
        assert entry.num_splits == 10

    def test_custom_num_splits(self):
        entry = DatasetEntry(metric="acc", split_type="pyg_mask", num_splits=5)
        assert entry.num_splits == 5


class TestDatasetRegistryIsolated:
    """Verify the registry base behaviour with a fresh instance."""

    def test_duplicate_registration_raises(self):
        reg = DatasetRegistry()
        reg.register("ds", DatasetEntry(metric="acc", split_type="npz_file"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register("ds", DatasetEntry(metric="acc", split_type="npz_file"))

    def test_contains(self):
        reg = DatasetRegistry()
        reg.register("ds", DatasetEntry(metric="acc", split_type="npz_file"))
        assert "ds" in reg
        assert "other" not in reg
