# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import random

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.nsd.nsd_layers import (
    DiagonalNSDConv,
    GeneralNSDConv,
    LowRankNSDConv,
    OrthogonalNSDConv,
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalNSDConv, {}),
        (GeneralNSDConv, {}),
        (OrthogonalNSDConv, {}),
        (GeneralNSDConv, {"use_attention": True}),
        (OrthogonalNSDConv, {"use_attention": True}),
        (OrthogonalNSDConv, {"orth_strategy": "fasth"}),
        (LowRankNSDConv, {"rank": 2}),
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
class TestNSDVariants:
    @pytest.fixture
    def setup(self, model_class, layer_kwargs):
        set_seed(42)
        # d=4, stalk_features=16 -> context_dim = 64
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        # x_feat must match context_dim for the map_generator
        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

        # Pass context_dim explicitly to the layer
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
        conv, x_feat, x_stalk, _ = setup
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(out, x_stalk)

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

    def test_permutation_invariance(self, setup):
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
        # SVD-based norm (low_rank) accumulates more FP error under permuted scatter
        tol = 1e-4 if isinstance(conv, LowRankNSDConv) else 1e-5
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=tol, rtol=tol)

    def test_restriction_map_initialization(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        diff = torch.abs(out - x_stalk).mean()
        assert diff < 1.0


def test_orthogonal_group_property():
    from sheaf_mpnn.utils import cayley

    set_seed(42)
    d, _stalk_features, _hidden_dim = 3, 16, 8
    # Test logic for Cayley transform independently
    params = torch.randn(10, (d * (d - 1)) // 2)
    caley_transform = cayley(params, d)
    identity = torch.eye(d).unsqueeze(0).repeat(10, 1, 1)
    torch.testing.assert_close(
        torch.matmul(caley_transform.transpose(-2, -1), caley_transform),
        identity,
        atol=1e-5,
        rtol=1e-5,
    )


def test_fasth_orthogonal_group_property():
    from sheaf_mpnn.utils import householder

    set_seed(42)
    d, _stalk_features, _hidden_dim = 3, 16, 8
    params = torch.randn(10, d * d)
    W = householder(params, d)
    identity = torch.eye(d).unsqueeze(0).repeat(10, 1, 1)
    torch.testing.assert_close(
        torch.matmul(W.transpose(-2, -1), W),
        identity,
        atol=1e-5,
        rtol=1e-5,
    )


def test_low_rank_map_rank_property():
    """Restriction maps F = A @ B^T have matrix rank <= r."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim, rank = 10, 16, 4, 8, 2
    context_dim = d * stalk_features
    conv = LowRankNSDConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, rank=rank
    )

    x_feat = torch.randn(num_nodes, context_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    t_diag, t = conv.get_map_products(x_feat, edge_index)
    for i in range(t_diag.shape[0]):
        assert torch.linalg.matrix_rank(t_diag[i]) <= rank
        assert torch.linalg.matrix_rank(t[i]) <= rank


def test_low_rank_invalid_rank():
    """LowRankNSDConv rejects non-positive rank."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankNSDConv(stalk_dim=4, in_channels=16, hidden_dim=8, rank=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_various_ranks(rank):
    """LowRankNSDConv produces valid outputs for different rank values."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
    context_dim = d * stalk_features
    conv = LowRankNSDConv(
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
    """Both strategies produce valid outputs in the full layer."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
    context_dim = d * stalk_features
    conv = OrthogonalNSDConv(
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
