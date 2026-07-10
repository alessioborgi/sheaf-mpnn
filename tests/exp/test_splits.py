# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for exp/splits.py -- 10-fold train/val/test split management."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from exp.data import DatasetInfo
from exp.splits import _apply_npz_split, _apply_pyg_mask_split, apply_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 30) -> Data:
    torch.manual_seed(0)
    return Data(
        x=torch.randn(n, 5),
        edge_index=torch.randint(0, n, (2, n * 2)),
        y=torch.randint(0, 3, (n,)),
        train_mask=torch.zeros(n, dtype=torch.bool),
        val_mask=torch.zeros(n, dtype=torch.bool),
        test_mask=torch.zeros(n, dtype=torch.bool),
    )


def _make_multicol_data(n: int = 30, n_splits: int = 10) -> Data:
    torch.manual_seed(0)
    return Data(
        x=torch.randn(n, 5),
        edge_index=torch.randint(0, n, (2, n * 2)),
        y=torch.randint(0, 3, (n,)),
        train_mask=torch.zeros(n, n_splits, dtype=torch.bool),
        val_mask=torch.zeros(n, n_splits, dtype=torch.bool),
        test_mask=torch.zeros(n, n_splits, dtype=torch.bool),
    )


def _make_info(split_type: str = "npz_file", name: str = "cora") -> DatasetInfo:
    return DatasetInfo(
        name=name,
        num_features=5,
        num_classes=3,
        num_splits=10,
        metric="acc",
        split_type=split_type,
    )


def _write_npz_split(directory: str, name: str, fold: int, n: int) -> tuple:
    """Write a synthetic split .npz and return (train, val, test) bool arrays."""
    train = np.zeros(n, dtype=bool)
    train[: int(n * 0.6)] = True
    val = np.zeros(n, dtype=bool)
    val[int(n * 0.6) : int(n * 0.8)] = True
    test = np.zeros(n, dtype=bool)
    test[int(n * 0.8) :] = True
    path = os.path.join(directory, f"{name}_split_0.6_0.2_{fold}.npz")
    np.savez(path, train_mask=train, val_mask=val, test_mask=test)
    return train, val, test


# ---------------------------------------------------------------------------
# _apply_npz_split
# ---------------------------------------------------------------------------


class TestApplyNpzSplit:
    @pytest.fixture
    def npz_setup(self):
        n = 30
        data = _make_data(n)
        with tempfile.TemporaryDirectory() as tmpdir:
            train, val, test = _write_npz_split(tmpdir, "cora", fold=0, n=n)
            yield tmpdir, data, train, val, test, n

    def test_train_mask_matches_npz(self, npz_setup):
        tmpdir, data, train, val, test, n = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        assert result.train_mask.sum().item() == int(train.sum())

    def test_val_mask_matches_npz(self, npz_setup):
        tmpdir, data, train, val, test, n = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        assert result.val_mask.sum().item() == int(val.sum())

    def test_test_mask_matches_npz(self, npz_setup):
        tmpdir, data, train, val, test, n = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        assert result.test_mask.sum().item() == int(test.sum())

    def test_output_masks_are_bool_tensors(self, npz_setup):
        tmpdir, data, *_ = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        assert result.train_mask.dtype == torch.bool
        assert result.val_mask.dtype == torch.bool
        assert result.test_mask.dtype == torch.bool

    def test_output_masks_are_1d(self, npz_setup):
        tmpdir, data, *_ = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        assert result.train_mask.dim() == 1
        assert result.val_mask.dim() == 1
        assert result.test_mask.dim() == 1

    def test_masks_are_mutually_exclusive(self, npz_setup):
        tmpdir, data, *_ = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        overlap = (result.train_mask & result.val_mask).any()
        assert not overlap

    def test_original_data_not_mutated(self, npz_setup):
        tmpdir, data, *_ = npz_setup
        original_train_sum = data.train_mask.sum().item()
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            _apply_npz_split(data, "cora", fold=0)
        assert data.train_mask.sum().item() == original_train_sum

    def test_returns_clone_not_same_object(self, npz_setup):
        tmpdir, data, *_ = npz_setup
        with patch("exp.splits._SPLITS_DIR", tmpdir):
            result = _apply_npz_split(data, "cora", fold=0)
        assert result is not data

    def test_different_folds_load_different_files(self):
        n = 20
        data = _make_data(n)
        with tempfile.TemporaryDirectory() as tmpdir:
            train0, _, _ = _write_npz_split(tmpdir, "cora", fold=0, n=n)
            # Fold 1: only first node in training
            train1 = np.zeros(n, dtype=bool)
            train1[0] = True
            val1 = np.zeros(n, dtype=bool)
            val1[1] = True
            test1 = np.zeros(n, dtype=bool)
            test1[2] = True
            np.savez(
                os.path.join(tmpdir, "cora_split_0.6_0.2_1.npz"),
                train_mask=train1,
                val_mask=val1,
                test_mask=test1,
            )
            with patch("exp.splits._SPLITS_DIR", tmpdir):
                result0 = _apply_npz_split(data, "cora", fold=0)
                result1 = _apply_npz_split(data, "cora", fold=1)
        assert result0.train_mask.sum().item() != result1.train_mask.sum().item()

    def test_missing_file_raises_runtime_error(self):
        data = _make_data()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("exp.splits._SPLITS_DIR", tmpdir),
            pytest.raises(RuntimeError, match="Failed to download"),
        ):
            _apply_npz_split(data, "nonexistent_dataset", fold=0)


# ---------------------------------------------------------------------------
# _apply_pyg_mask_split
# ---------------------------------------------------------------------------


class TestApplyPygMaskSplit:
    def test_selects_correct_fold_column(self):
        n, n_splits = 30, 10
        data = _make_multicol_data(n, n_splits)
        data.train_mask[:, 3] = True  # Fold 3 is all-True
        result = _apply_pyg_mask_split(data, fold=3)
        assert result.train_mask.all()

    def test_output_masks_are_1d(self):
        data = _make_multicol_data()
        result = _apply_pyg_mask_split(data, fold=0)
        assert result.train_mask.dim() == 1
        assert result.val_mask.dim() == 1
        assert result.test_mask.dim() == 1

    def test_output_has_correct_length(self):
        n = 25
        data = _make_multicol_data(n=n, n_splits=10)
        result = _apply_pyg_mask_split(data, fold=0)
        assert result.train_mask.shape == (n,)

    def test_fold_wraps_when_exceeds_n_splits(self):
        n, n_splits = 30, 10
        data = _make_multicol_data(n, n_splits)
        data.val_mask[:, 2] = True  # fold 12 % 10 == 2
        result = _apply_pyg_mask_split(data, fold=12)
        assert result.val_mask.all()

    def test_single_column_mask_returned_unchanged(self):
        n = 30
        data = _make_data(n)
        data.train_mask[:] = True
        data.val_mask[:10] = True
        result = _apply_pyg_mask_split(data, fold=5)
        assert result.train_mask.dim() == 1
        assert result.train_mask.all()

    def test_original_data_not_mutated(self):
        data = _make_multicol_data()
        original_shape = data.train_mask.shape
        _apply_pyg_mask_split(data, fold=0)
        assert data.train_mask.shape == original_shape

    def test_returns_clone_not_same_object(self):
        data = _make_multicol_data()
        result = _apply_pyg_mask_split(data, fold=0)
        assert result is not data

    def test_different_folds_select_different_columns(self):
        n, n_splits = 30, 10
        data = _make_multicol_data(n, n_splits)
        # Mark fold 0 col as train, fold 1 col as val
        data.train_mask[:, 0] = True
        data.val_mask[:, 1] = True
        result0 = _apply_pyg_mask_split(data, fold=0)
        result1 = _apply_pyg_mask_split(data, fold=1)
        assert result0.train_mask.all()
        assert result1.val_mask.all()
        assert not result1.train_mask.all()


# ---------------------------------------------------------------------------
# apply_split (public API / dispatch)
# ---------------------------------------------------------------------------


class TestApplySplitDispatch:
    def test_npz_file_split_type_dispatches_correctly(self):
        n = 20
        data = _make_data(n)
        info = _make_info(split_type="npz_file", name="cora")
        with tempfile.TemporaryDirectory() as tmpdir:
            train, val, test = _write_npz_split(tmpdir, "cora", fold=0, n=n)
            with patch("exp.splits._SPLITS_DIR", tmpdir):
                result = apply_split(data, info, fold=0)
        assert result.train_mask.sum().item() == int(train.sum())
        assert result.val_mask.sum().item() == int(val.sum())
        assert result.test_mask.sum().item() == int(test.sum())

    def test_pyg_mask_split_type_dispatches_correctly(self):
        n, n_splits = 20, 10
        data = _make_multicol_data(n, n_splits)
        data.test_mask[:, 4] = True
        info = _make_info(split_type="pyg_mask", name="amazon_ratings")
        result = apply_split(data, info, fold=4)
        assert result.test_mask.all()
        assert result.test_mask.dim() == 1

    def test_result_masks_are_1d_for_npz_split(self):
        n = 20
        data = _make_data(n)
        info = _make_info(split_type="npz_file", name="texas")
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_npz_split(tmpdir, "texas", fold=0, n=n)
            with patch("exp.splits._SPLITS_DIR", tmpdir):
                result = apply_split(data, info, fold=0)
        assert result.train_mask.dim() == 1

    def test_result_masks_are_1d_for_pyg_mask(self):
        data = _make_multicol_data()
        info = _make_info(split_type="pyg_mask", name="roman_empire")
        result = apply_split(data, info, fold=0)
        assert result.train_mask.dim() == 1

    @pytest.mark.parametrize("fold", [0, 3, 7, 9])
    def test_multiple_folds_npz(self, fold):
        n = 20
        data = _make_data(n)
        info = _make_info(split_type="npz_file", name="cora")
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_npz_split(tmpdir, "cora", fold=fold, n=n)
            with patch("exp.splits._SPLITS_DIR", tmpdir):
                result = apply_split(data, info, fold=fold)
        assert result.train_mask.dtype == torch.bool

    @pytest.mark.parametrize("fold", [0, 3, 7, 9])
    def test_multiple_folds_pyg_mask(self, fold):
        data = _make_multicol_data(n_splits=10)
        info = _make_info(split_type="pyg_mask")
        result = apply_split(data, info, fold=fold)
        assert result.train_mask.dim() == 1
