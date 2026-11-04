# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.sheaf_adj.sheaf_adj_layers import (
    DiagonalHetSheafAdjConv,
    GeneralHetSheafAdjConv,
    LowRankHetSheafAdjConv,
    OrthogonalHetSheafAdjConv,
)
from tests.testing import set_seed


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalHetSheafAdjConv, {}),
        (GeneralHetSheafAdjConv, {}),
        (OrthogonalHetSheafAdjConv, {}),
        (GeneralHetSheafAdjConv, {"use_attention": True}),
        (OrthogonalHetSheafAdjConv, {"use_attention": True}),
        (OrthogonalHetSheafAdjConv, {"orth_strategy": "fasth"}),
        (LowRankHetSheafAdjConv, {"rank": 2}),
    ],
    ids=[
        "diagonal",
        "general",
        "orthogonal",
        "general_attention",
        "orthogonal_attention",
        "orthogonal_fasth",
        "low_rank",
    ],
)
class TestHetSheafAdjVariants:
    @pytest.fixture
    def setup(self, model_class, layer_kwargs):
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

        conv = model_class(
            d, stalk_features, hidden_dim, context_dim=context_dim, **layer_kwargs
        )
        return conv, x_feat, x_stalk, edge_index

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_isolated_node_stability(self, setup):
        # SheafAdj adjacency does not cancel self-loop terms with near-singular maps.
        # Check finitude; exact recovery requires alpha=0 (test_alpha_zero_identity).
        conv, x_feat, x_stalk, _ = setup
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()

    def test_alpha_zero_identity(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        conv.alpha.data.zero_()
        out = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(out, x_stalk)

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        assert conv.W2.grad is not None and conv.W2.grad.abs().sum() > 0
        for param in conv.map_generator.parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0

    def test_permutation_invariance(self, setup, model_class):
        conv, x_feat, x_stalk, edge_index = setup
        conv.eval()
        out_orig = conv(x_feat, x_stalk, edge_index)

        perm = torch.randperm(x_feat.size(0))
        rev_perm = torch.argsort(perm)
        row, col = edge_index
        mapping = torch.zeros(x_feat.size(0), dtype=torch.long)
        mapping[perm] = torch.arange(x_feat.size(0))
        edge_index_p = torch.stack([mapping[row], mapping[col]], dim=0)

        out_perm = conv(x_feat[perm], x_stalk[perm], edge_index_p)
        tol = 1e-4 if model_class is LowRankHetSheafAdjConv else 1e-5
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=tol, rtol=tol)

    def test_restriction_map_initialization(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        diff = torch.abs(out - x_stalk).mean()
        assert diff < 1.0


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalHetSheafAdjConv, {}),
        (GeneralHetSheafAdjConv, {}),
        (OrthogonalHetSheafAdjConv, {}),
    ],
    ids=["diagonal", "general", "orthogonal"],
)
def test_with_node_types(model_class, layer_kwargs):
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 8, 8, 2, 4
    num_node_types = 3
    context_dim = d * stalk_features

    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    node_types = torch.randint(0, num_node_types, (num_nodes,))

    conv = model_class(
        d,
        stalk_features,
        hidden_dim,
        num_node_types=num_node_types,
        use_node_types=True,
        context_dim=context_dim,
        **layer_kwargs,
    )
    out = conv(x_feat, x_stalk, edge_index, node_types=node_types)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


def test_low_rank_invalid_rank_het():
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankHetSheafAdjConv(stalk_dim=4, in_channels=16, hidden_dim=8, rank=0)
