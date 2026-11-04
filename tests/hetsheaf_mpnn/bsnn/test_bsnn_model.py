# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph, to_undirected

from hetsheaf_mpnn.bsnn import BSNNVariant, HetBSNNModel

NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4

# HetBSNN has no heterogeneous low-rank variational learner; LOW_RANK is
# rejected at construction (see test_low_rank_is_rejected).
HET_VARIANTS = [v for v in BSNNVariant if v is not BSNNVariant.LOW_RANK]


def _graph():
    # Both orientations of every undirected edge, as the BSNN sampler expects.
    return to_undirected(erdos_renyi_graph(NUM_NODES, edge_prob=0.4))


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


@pytest.mark.parametrize("variant", HET_VARIANTS)
def test_construction_and_forward(variant):
    torch.manual_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = HetBSNNModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=2,
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
    model = HetBSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        use_features=True,
        use_node_types=False,
        use_edge_types=False,
    )
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


@pytest.mark.parametrize("variant", HET_VARIANTS)
def test_kl_loss(variant):
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetBSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        variant=variant,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
    )
    assert model.kl_loss().item() == 0.0
    model(x, edge_index, nt, et)
    kl = model.kl_loss()
    assert kl.dim() == 0
    assert torch.isfinite(kl)


def test_elbo_backprop():
    """Gradients flow through both the CE and the KL term of the ELBO."""
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    y = torch.randint(0, OUT_CHANNELS, (NUM_NODES,))
    model = HetBSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    loss = torch.nn.functional.cross_entropy(out, y) + 0.1 * model.kl_loss()
    loss.backward()
    assert model.encoder.weight.grad is not None
    assert model.sheaf_learner.lin_mean.weight.grad is not None


def test_predict_proba():
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetBSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        ensemble_size=2,
    )
    probs = model.predict_proba(x, edge_index, nt, et)
    assert probs.shape == (2, NUM_NODES, OUT_CHANNELS)
    assert torch.allclose(probs.sum(-1), torch.ones(2, NUM_NODES))
    probs5 = model.predict_proba(x, edge_index, nt, et, num_samples=5)
    assert probs5.shape == (5, NUM_NODES, OUT_CHANNELS)


def test_empty_graph():
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    model = HetBSNNModel(IN_CHANNELS, OUT_CHANNELS)
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert model.kl_loss().item() == 0.0


def test_het_learner_input_width():
    """The variational heads widen by exactly the enabled type context."""
    model = HetBSNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=8,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=True,
        use_edge_types=True,
    )
    context_dim = 2 * 8
    expected = 2 * context_dim + 2 * NUM_NODE_TYPES + NUM_EDGE_TYPES
    assert model.sheaf_learner.lin_mean.in_features == expected
    assert model.sheaf_learner.lin_scale.in_features == expected


def test_invalid_args():
    with pytest.raises(ValueError, match="stalk_dim"):
        HetBSNNModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)
    with pytest.raises(ValueError, match="orthogonal BSNN"):
        HetBSNNModel(
            IN_CHANNELS, OUT_CHANNELS, stalk_dim=1, variant=BSNNVariant.ORTHOGONAL
        )
    with pytest.raises(ValueError, match="low_rank"):
        HetBSNNModel(IN_CHANNELS, OUT_CHANNELS, variant=BSNNVariant.LOW_RANK)
    with pytest.raises(ValueError, match="at least one"):
        HetBSNNModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)
    with pytest.raises(ValueError, match="num_node_types"):
        HetBSNNModel(IN_CHANNELS, OUT_CHANNELS, use_node_types=True)
    with pytest.raises(ValueError, match="ensemble_size"):
        HetBSNNModel(IN_CHANNELS, OUT_CHANNELS, ensemble_size=0)


def test_reset_parameters():
    model = HetBSNNModel(IN_CHANNELS, OUT_CHANNELS, num_node_types=2, num_edge_types=2)
    model.reset_parameters()
