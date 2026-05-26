# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

import pytest
import torch

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.nsd.nsd_model import NSDModel, NSDVariant


def test_base_conv_stalk_transform():
    # Covers src/sheaf_mpnn/base_conv.py: _apply_stalk_transform
    conv = BaseSheafConv(stalk_dim=2, in_channels=4, hidden_dim=4, add_self_loops=False)
    x = torch.randn(5, 2, 4)
    out = conv._apply_stalk_transform(x)
    assert out.shape == x.shape


def test_nsd_model_reset_parameters():
    model = NSDModel(
        in_channels=8, out_channels=3, stalk_dim=2, hidden_dim=4, num_layers=1
    )
    model.reset_parameters()
    # Should not raise


def test_nsd_no_self_loops():
    # Covers src/sheaf_mpnn/nsd/nsd_layers.py:94
    model = NSDModel(
        in_channels=8,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        add_self_loops=False,
    )
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (10, 3)


def test_nsd_variant_ortho_generator():
    # Trying to cover nsd_layers.py:94
    model = NSDModel(
        in_channels=8,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        variant=NSDVariant.ORTHOGONAL,
    )
    # Just running a forward pass might trigger map generation
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).repeat(1, 5)
    out = model(x, edge_index)
    assert out.shape == (10, 3)


def test_load_dataset_unknown():
    from exp.data import load_dataset

    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("unknown_ds")
