# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.sheaf_attn.sheaf_attn_model import (
    HetSheafAttnModel,
    HetSheafAttnVariant,
)

NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


@pytest.mark.parametrize("variant", list(HetSheafAttnVariant))
def test_construction_and_forward(variant):
    torch.manual_seed(42)
    edge_index = _graph()
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = HetSheafAttnModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        variant=variant,
        num_heads=2,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


def test_forward_without_types():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetSheafAttnModel(IN_CHANNELS, OUT_CHANNELS)
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_jknet():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetSheafAttnModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=3,
        num_heads=2,
        jknet=True,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
    )
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_gradient_backprop():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    model = HetSheafAttnModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
    )
    model(x, edge_index, nt, et).sum().backward()
    assert model.encoder.weight.grad is not None


def test_invalid_construction():
    with pytest.raises(ValueError):
        HetSheafAttnModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)
    with pytest.raises(ValueError):
        HetSheafAttnModel(IN_CHANNELS, OUT_CHANNELS, num_heads=0)


def test_invalid_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        HetSheafAttnModel(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)


def test_invalid_num_layers():
    with pytest.raises(ValueError, match="at least one"):
        HetSheafAttnModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)


def test_reset_parameters():
    model = HetSheafAttnModel(
        IN_CHANNELS, OUT_CHANNELS, num_node_types=2, num_edge_types=2
    )
    model.reset_parameters()


def test_use_features_false():
    """Verify that HetSheafAttnModel works with use_features=False.

    Ensures it works when node types are provided but features are disabled.
    """
    edge_index = _graph()
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = HetSheafAttnModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=False,
        use_node_types=True,
        use_edge_types=True,
    )

    # Check that layers have correct context dimension (only node types)
    for layer in model.layers:
        assert layer._attn_ctx_dim == NUM_NODE_TYPES
        assert isinstance(layer.attn_projection, torch.nn.Linear)
        assert layer.attn_projection.in_features == NUM_NODE_TYPES
        assert layer._map_gen_in == 2 * NUM_NODE_TYPES + NUM_EDGE_TYPES

    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


def test_all_context_false_raises():
    """Verify that disabling all context flags raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        HetSheafAttnModel(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            use_features=False,
            use_node_types=False,
            use_edge_types=False,
            ensemble=False,
        )
