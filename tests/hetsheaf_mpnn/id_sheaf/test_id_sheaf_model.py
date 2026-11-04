# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.id_sheaf import HetIdentitySheafModel

NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


def test_construction_and_forward():
    torch.manual_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = HetIdentitySheafModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


def test_forward_without_types():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS)
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_types_do_not_change_output():
    """Identity maps are type-blind: outputs match with and without types."""
    torch.manual_seed(0)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS)
    model.eval()
    out_plain = model(x, edge_index)
    out_typed = model(x, edge_index, nt, et)
    assert torch.allclose(out_plain, out_typed)


def test_jknet():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetIdentitySheafModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=3,
        jknet=True,
        normalize_output=True,
    )
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_gradient_backprop():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS)
    out = model(x, edge_index)
    out.sum().backward()
    assert model.encoder.weight.grad is not None


def test_invalid_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim"):
        HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)


def test_invalid_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)


def test_invalid_num_layers():
    with pytest.raises(ValueError, match="at least one"):
        HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)


def test_reset_parameters():
    model = HetIdentitySheafModel(IN_CHANNELS, OUT_CHANNELS)
    model.reset_parameters()
