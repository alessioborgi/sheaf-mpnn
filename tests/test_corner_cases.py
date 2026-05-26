# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

import os
from unittest.mock import patch

import pytest
import torch

from sheaf_mpnn.nsd.nsd_layers import (
    DiagonalNSDConv,
    GeneralNSDConv,
    OrthogonalNSDConv,
)
from sheaf_mpnn.utils import setup_torch


def test_setup_torch_mps(monkeypatch):
    """Cover the MPS fallback branch in setup_torch."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class MockMPS:
        def is_available(self):
            return True

    monkeypatch.setattr(torch.backends, "mps", MockMPS(), raising=False)

    # We need to ensure print doesn't fail or we can mock it
    setup_torch(precision="high", seed=42)
    # Check if the fallback happened (is_available should be patched to False)
    assert torch.backends.mps.is_available() is False


def test_nsd_layers_no_self_loops():
    """Verify NSD layers work correctly without self-loops."""
    num_nodes, d, f = 4, 2, 4
    x_feat = torch.randn(num_nodes, d * f)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    for cls in [DiagonalNSDConv, GeneralNSDConv, OrthogonalNSDConv]:
        conv = cls(d, f, hidden_dim=8, add_self_loops=False)
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape


def test_nsd_alpha_initialization():
    """Verify alpha is correctly initialized and is a learnable parameter."""
    conv = GeneralNSDConv(stalk_dim=2, in_channels=4, hidden_dim=8, alpha=0.5)
    assert isinstance(conv.alpha, torch.nn.Parameter)
    assert conv.alpha.item() == pytest.approx(0.5)


def test_orthogonal_nsd_clamping():
    """Verify clamp_val is used in OrthogonalNSDConv."""
    num_nodes, d, f = 3, 2, 2
    x_feat = torch.randn(num_nodes, d * f)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    # Use a very small clamp_val and large params to see if it has an effect
    conv = OrthogonalNSDConv(d, f, hidden_dim=8, clamp_val=0.01)
    # Forward pass should still work and be finite
    out = conv(x_feat, x_stalk, edge_index)
    assert torch.isfinite(out).all()


def test_base_conv_reset_parameters_no_map_generator():
    """Verify reset_parameters works even if map_generator is missing (safety check)."""
    from sheaf_mpnn.base_conv import BaseSheafConv

    conv = BaseSheafConv(stalk_dim=2, in_channels=4, hidden_dim=4)
    # Should not raise even if it doesn't have map_generator
    conv.reset_parameters()


def test_diagonal_nsd_message_shapes():
    """Explicitly check message() shapes for DiagonalNSDConv."""
    conv = DiagonalNSDConv(stalk_dim=2, in_channels=4, hidden_dim=8)
    x_i = torch.randn(5, 2, 4)
    x_j = torch.randn(5, 2, 4)
    t_diag = torch.randn(5, 2)
    t = torch.randn(5, 2)
    msg = conv.message(x_i, x_j, t_diag, t)
    assert msg.shape == (5, 2, 4)


def test_nsd_attention_variants():
    """Cover the use_attention=True branches in NSD layers."""
    num_nodes, d, f = 4, 2, 2
    x_feat = torch.randn(num_nodes, d * f)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    for cls in [GeneralNSDConv, OrthogonalNSDConv]:
        conv = cls(d, f, hidden_dim=8, use_attention=True)
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape


def test_nsd_variant_enum_properties():
    """Cover all branches of NSDVariant properties."""
    from sheaf_mpnn.nsd.nsd_model import NSDVariant

    for v in NSDVariant:
        assert v.layer_class is not None
        assert isinstance(v.layer_kwargs, dict)


def test_attention_cayley_explicit_device_dtype():
    """Cover the branches where device and dtype are NOT None in attention_cayley."""
    from sheaf_mpnn.utils import attention_cayley

    d = 2
    raw = torch.randn(1, d * d)
    # Pass explicit device and dtype
    W = attention_cayley(raw, d, device=raw.device, dtype=raw.dtype)
    assert W.shape == (1, d, d)


def test_setup_torch_cpu(monkeypatch):
    """Cover the CPU branch in setup_torch."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Ensure mps is not available
    class MockMPS:
        def is_available(self):
            return False

    monkeypatch.setattr(torch.backends, "mps", MockMPS(), raising=False)

    setup_torch(precision="high", seed=42)


def test_attention_cayley_default_device_dtype():
    """Cover the branches where device and dtype ARE None in attention_cayley."""
    from sheaf_mpnn.utils import attention_cayley

    d = 2
    raw = torch.randn(1, d * d)
    # Pass None (or nothing)
    W = attention_cayley(raw, d, device=None, dtype=None)
    assert W.shape == (1, d, d)


def test_load_dataset_assertion_error():
    """Trigger AssertionError in load_dataset for unhandled loader kind."""
    from exp.data import _LOADER, load_dataset

    with patch.dict(_LOADER, {"fake_ds": ("unknown_kind", "FakeKey")}):
        with pytest.raises(AssertionError, match="Unhandled loader kind"):
            load_dataset("fake_ds")


def test_download_split_success(tmp_path):
    """Cover the success print in _download_split."""
    from exp.splits import _download_split

    path = os.path.join(tmp_path, "fake_split.npz")
    with patch("exp.splits.urllib.request.urlretrieve") as mock_retrieve:
        # Mock successful download (create the file)
        def side_effect(url, dst):
            with open(dst, "w") as f:
                f.write("fake data")

        mock_retrieve.side_effect = side_effect
        _download_split("cora", 0, path)
    assert os.path.exists(path)


def test_download_split_failure_removes_file(tmp_path):
    """Cover the file removal in _download_split failure branch."""
    from exp.splits import _download_split

    path = os.path.join(tmp_path, "fake_split.npz")
    with patch("exp.splits.urllib.request.urlretrieve") as mock_retrieve:
        # Mock failed download that creates a partial/empty file
        def side_effect(url, dst):
            with open(dst, "w") as f:
                f.write("partial")
            raise Exception("Download failed")

        mock_retrieve.side_effect = side_effect
        with pytest.raises(RuntimeError, match="Failed to download"):
            _download_split("cora", 0, path)
    assert not os.path.exists(path)


def test_nsd_datamodule_test_loader():
    """Cover test_dataloader in SheafDataModule."""
    from torch_geometric.data import Data

    from exp.data import DatasetInfo, SheafDataModule

    dm = SheafDataModule("cora", root="/tmp/fake")
    data = Data(
        x=torch.randn(10, 4),
        y=torch.randint(0, 2, (10,)),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    data.train_mask = torch.ones(10, dtype=torch.bool)
    data.val_mask = torch.ones(10, dtype=torch.bool)
    data.test_mask = torch.ones(10, dtype=torch.bool)
    info = DatasetInfo("cora", 4, 2, 10, "acc", "pyg_mask")

    with (
        patch("exp.data.load_dataset", return_value=(data, info)),
        patch("exp.splits.apply_split", return_value=data),
    ):
        dm.setup()
        assert dm.val_dataloader() is not None
        loader = dm.test_dataloader()
        assert loader is not None
        batch = next(iter(loader))
        assert batch.num_nodes == 10
