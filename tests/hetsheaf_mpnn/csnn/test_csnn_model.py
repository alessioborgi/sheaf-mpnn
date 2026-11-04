# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.csnn import CSNNVariant, HetCSNNModel

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


@pytest.mark.parametrize("variant", list(CSNNVariant))
def test_construction_and_forward_with_node_types(variant):
    torch.manual_seed(42)
    edge_index = _graph()
    nt, _ = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = HetCSNNModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=8,
        num_layers=2,
        variant=variant,
        num_node_types=NUM_NODE_TYPES,
        use_node_types=True,
    )
    out = model(x, edge_index, nt)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


def test_forward_without_types():
    """Model works without type tensors (features-only mode)."""
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetCSNNModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=2, hidden_dim=8)
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_edge_types_with_edge_feature_backbone():
    """Edge-type one-hots reach the learner through an NNConv backbone."""
    torch.manual_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetCSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=8,
        num_layers=1,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
        gnn_layers=1,
        gnn_default=0,
        gnn_type="NNConv",
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_edge_types_rejected_on_default_backbone():
    """The default GraphSAGE preset ignores edge_attr, so use_edge_types must
    fail loudly instead of silently dropping the edge types.
    """
    with pytest.raises(ValueError, match="use_edge_types"):
        HetCSNNModel(
            IN_CHANNELS,
            OUT_CHANNELS,
            stalk_dim=2,
            num_edge_types=NUM_EDGE_TYPES,
            use_edge_types=True,
        )


def test_node_types_change_output():
    """Node-type one-hots feed the sheaf learners, so outputs differ."""
    torch.manual_seed(0)
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetCSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=8,
        num_node_types=2,
        use_node_types=True,
    )
    model.eval()
    out_a = model(x, edge_index, torch.zeros(NUM_NODES, dtype=torch.long))
    out_b = model(x, edge_index, torch.ones(NUM_NODES, dtype=torch.long))
    assert not torch.allclose(out_a, out_b)


def test_gradient_backprop():
    edge_index = _graph()
    nt, _ = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetCSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=8,
        num_node_types=NUM_NODE_TYPES,
        use_node_types=True,
    )
    out = model(x, edge_index, nt)
    out.sum().backward()
    assert model.encoder.weight.grad is not None


def test_pe_and_node_types_compose():
    """A user positional encoding coexists with the node-type widening."""
    edge_index = _graph()
    nt, _ = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    pe = torch.randn(NUM_NODES, 5)
    model = HetCSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=8,
        num_node_types=NUM_NODE_TYPES,
        use_node_types=True,
        pe_size=5,
    )
    out = model(x, edge_index, nt, pe=pe)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    with pytest.raises(ValueError, match="pe has width"):
        model(x, edge_index, nt, pe=torch.randn(NUM_NODES, 4))


def test_invalid_args():
    with pytest.raises(ValueError, match="stalk_dim"):
        HetCSNNModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=1)
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        HetCSNNModel(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)
    with pytest.raises(ValueError, match="at least one"):
        HetCSNNModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)
    with pytest.raises(ValueError, match="num_node_types"):
        HetCSNNModel(IN_CHANNELS, OUT_CHANNELS, use_node_types=True)


def test_reset_parameters():
    model = HetCSNNModel(
        IN_CHANNELS, OUT_CHANNELS, num_node_types=2, use_node_types=True
    )
    model.reset_parameters()
