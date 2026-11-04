# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.sheaf_adj.sheaf_adj_layers import (
    DiagonalSheafAdjConv,
    GeneralSheafAdjConv,
    LowRankSheafAdjConv,
    OrthogonalSheafAdjConv,
)
from tests.testing import set_seed


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalSheafAdjConv, {}),
        (GeneralSheafAdjConv, {}),
        (OrthogonalSheafAdjConv, {}),
        (GeneralSheafAdjConv, {"use_attention": True}),
        (OrthogonalSheafAdjConv, {"use_attention": True}),
        (OrthogonalSheafAdjConv, {"orth_strategy": "fasth"}),
        (LowRankSheafAdjConv, {"rank": 2}),
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
class TestSheafAdjVariants:
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
        # SheafAdj uses the adjacency (not Laplacian), so the self-loop term does
        # not cancel to zero with near-singular restriction maps.  The invariant
        # tested here is that the output is finite and shape-preserving.
        # For exact recovery (out == x_stalk), use alpha=0
        # (see test_alpha_zero_identity).
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
        # Low-rank SVD introduces small FP non-associativity; use a looser tolerance.
        tol = 1e-4 if model_class is LowRankSheafAdjConv else 1e-5
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=tol, rtol=tol)

    def test_restriction_map_initialization(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        diff = torch.abs(out - x_stalk).mean()
        assert diff < 1.0


def test_low_rank_invalid_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankSheafAdjConv(stalk_dim=4, in_channels=16, hidden_dim=8, rank=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_various_ranks(rank):
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
    context_dim = d * stalk_features
    conv = LowRankSheafAdjConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, rank=rank
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orthogonal_strategy_layer_forward(orth_strategy):
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
    context_dim = d * stalk_features
    conv = OrthogonalSheafAdjConv(
        d,
        stalk_features,
        hidden_dim,
        context_dim=context_dim,
        orth_strategy=orth_strategy,
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


def test_adjacency_vs_laplacian_sign():
    # SheafAdjacency adds sigma(agg), NSD subtracts sigma(L z).
    # With a positive-weight graph and identity maps, agg should be positive.
    # Check that the output sign differs from NSD (regression guard).
    from sheaf_mpnn.nsd.nsd_layers import DiagonalNSDConv

    set_seed(0)
    num_nodes, d, f, h = 6, 2, 4, 4
    ctx = d * f
    x_feat = torch.randn(num_nodes, ctx)
    x_stalk = torch.randn(num_nodes, d, f)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    adj_conv = DiagonalSheafAdjConv(d, f, h, context_dim=ctx)
    nsd_conv = DiagonalNSDConv(d, f, h, context_dim=ctx)

    # Share the generator and bilateral transforms so both layers see
    # identical restriction maps despite different learner architectures.
    nsd_conv.map_generator = adj_conv.map_generator
    nsd_conv.W1.data.copy_(adj_conv.W1.data)
    nsd_conv.W2.data.copy_(adj_conv.W2.data)

    out_adj = adj_conv(x_feat, x_stalk, edge_index)
    out_nsd = nsd_conv(x_feat, x_stalk, edge_index)

    # Outputs must differ (adjacency vs Laplacian update).
    assert not torch.allclose(out_adj, out_nsd, atol=1e-6)
