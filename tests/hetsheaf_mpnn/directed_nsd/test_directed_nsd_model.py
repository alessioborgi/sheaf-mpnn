# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.directed_nsd import HetDirectedNSDModel, HetDirectedNSDVariant

NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _graph():
    # Directed: no symmetrization, as produced by erdos_renyi (directed=True).
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4, directed=True)


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


@pytest.mark.parametrize("variant", list(HetDirectedNSDVariant))
def test_construction_and_forward(variant):
    torch.manual_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = HetDirectedNSDModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        variant=variant,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


def test_forward_without_types():
    """Model works without type tensors (features-only mode)."""
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetDirectedNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        use_features=True,
        use_node_types=False,
        use_edge_types=False,
    )
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_ensemble_model():
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    model = HetDirectedNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        ensemble=True,
    )
    out = model(x, edge_index, None, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_gradient_backprop():
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetDirectedNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    out.sum().backward()
    assert model.encoder.weight.grad is not None


def test_direction_sensitivity():
    """Reversing all edges changes the output (directional aggregation)."""
    torch.manual_seed(7)
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetDirectedNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        variant=HetDirectedNSDVariant.GENERAL,
        alpha=0.9,
    )
    model.eval()
    out_fwd = model(x, edge_index)
    out_rev = model(x, edge_index.flip(0))
    assert not torch.allclose(out_fwd, out_rev)


def test_jknet():
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetDirectedNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=3,
        jknet=True,
        normalize_output=True,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_invalid_args():
    with pytest.raises(ValueError, match="stalk_dim"):
        HetDirectedNSDModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        HetDirectedNSDModel(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)
    with pytest.raises(ValueError, match="at least one"):
        HetDirectedNSDModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)
    with pytest.raises(ValueError, match="orth_strategy"):
        HetDirectedNSDModel(
            IN_CHANNELS,
            OUT_CHANNELS,
            variant=HetDirectedNSDVariant.ORTHOGONAL,
            orth_strategy="fasthpp",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="rank"):
        HetDirectedNSDModel(
            IN_CHANNELS, OUT_CHANNELS, variant=HetDirectedNSDVariant.LOW_RANK, rank=0
        )


def test_reset_parameters():
    model = HetDirectedNSDModel(
        IN_CHANNELS, OUT_CHANNELS, num_node_types=2, num_edge_types=2
    )
    model.reset_parameters()
