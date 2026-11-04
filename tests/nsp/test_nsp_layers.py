# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.nsp.nsp_layers import (
    DiagonalNSPConv,
    GeneralNSPConv,
    LowRankNSPConv,
    OrthogonalNSPConv,
)
from sheaf_mpnn.utils import attention_cayley, cayley
from tests.testing import set_seed


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalNSPConv, {}),
        (GeneralNSPConv, {}),
        (OrthogonalNSPConv, {}),
        (GeneralNSPConv, {"use_attention": True}),
        (OrthogonalNSPConv, {"use_attention": True}),
        (LowRankNSPConv, {"rank": 2}),
    ],
    ids=[
        "diagonal",
        "general",
        "orthogonal",
        "general_attention",
        "orthogonal_attention",
        "low_rank",
    ],
)
class TestNSPVariants:
    @pytest.fixture
    def setup(self, model_class, layer_kwargs):
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        # Augmented stalk: [N, 2d, f]
        x_stalk = torch.randn(num_nodes, 2 * d, stalk_features)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

        conv = model_class(
            d, stalk_features, hidden_dim, context_dim=context_dim, **layer_kwargs
        )
        return conv, x_feat, x_stalk, edge_index, d, stalk_features

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape  # [N, 2d, f]
        assert not torch.isnan(out).any()

    def test_history_update(self, setup):
        # After one step, the history block must equal the previous current block.
        conv, x_feat, x_stalk, edge_index, d, f = setup
        x_curr_before = x_stalk[:, :d, :].clone()
        out = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(out[:, d:, :], x_curr_before)

    def test_isolated_nodes_recurrence(self, setup):
        # No edges -> laplacian = 0 -> sigma(0) = 0 -> new_curr = 2*x_curr - x_hist.
        conv, x_feat, x_stalk, _, d, f = setup
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x_curr = x_stalk[:, :d, :]
        x_hist = x_stalk[:, d:, :]
        out = conv(x_feat, x_stalk, edge_index)
        expected_curr = 2 * x_curr - x_hist
        torch.testing.assert_close(out[:, :d, :], expected_curr)
        torch.testing.assert_close(out[:, d:, :], x_curr)

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        out = conv(x_feat, x_stalk, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        assert conv.W2.grad is not None and conv.W2.grad.abs().sum() > 0
        for param in conv.map_generator.parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0

    def test_permutation_invariance(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
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
        tol = 1e-4 if isinstance(conv, LowRankNSPConv) else 1e-5
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=tol, rtol=tol)

    def test_numerical_stability(self, setup):
        conv, x_feat, x_stalk, edge_index, d, f = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert torch.isfinite(out).all()


def test_orthogonal_group_property():
    set_seed(42)
    d, _stalk_features, _hidden_dim = 3, 16, 8
    params = torch.randn(10, (d * (d - 1)) // 2)
    cayley_transform = cayley(params, d)
    identity = torch.eye(d).unsqueeze(0).repeat(10, 1, 1)
    torch.testing.assert_close(
        torch.matmul(cayley_transform.transpose(-2, -1), cayley_transform),
        identity,
        atol=1e-5,
        rtol=1e-5,
    )


def test_attention_cayley_near_orthogonal():
    # attention_cayley uses the skew-symmetric part of the attention map, so W^T W = I.
    set_seed(42)
    d, _stalk_features, _hidden_dim = 3, 16, 8
    raw = torch.randn(10, d * d)
    W = attention_cayley(raw, d, device=raw.device, dtype=raw.dtype)
    identity = torch.eye(d).unsqueeze(0).repeat(10, 1, 1)
    torch.testing.assert_close(
        torch.matmul(W.transpose(-2, -1), W), identity, atol=1e-5, rtol=1e-5
    )


def test_attention_general_map_shape():
    # Verify that the attention formula I - softmax(*) produces [E, d, d] maps.
    set_seed(42)
    d, stalk_features, hidden_dim = 4, 8, 16
    context_dim = d * stalk_features
    conv = GeneralNSPConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, use_attention=True
    )
    x_feat = torch.randn(5, context_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    t_diag, t = conv.get_map_products(x_feat, edge_index)
    assert t_diag.shape == (3, d, d)
    assert t.shape == (3, d, d)
    assert torch.isfinite(t_diag).all()
    assert torch.isfinite(t).all()


def test_low_rank_map_rank_property():
    """Restriction maps F = left @ right^T have matrix rank <= r."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim, rank = 10, 16, 4, 8, 2
    context_dim = d * stalk_features
    conv = LowRankNSPConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, rank=rank
    )

    x_feat = torch.randn(num_nodes, context_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    self_map, cross_map = conv.get_map_products(x_feat, edge_index)
    for i in range(self_map.shape[0]):
        assert torch.linalg.matrix_rank(self_map[i]) <= rank
        assert torch.linalg.matrix_rank(cross_map[i]) <= rank


def test_low_rank_invalid_rank():
    """LowRankNSPConv rejects non-positive rank."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankNSPConv(stalk_dim=4, in_channels=16, hidden_dim=8, rank=0)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_various_ranks(rank):
    """LowRankNSPConv produces valid augmented-state outputs for different ranks."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
    context_dim = d * stalk_features
    conv = LowRankNSPConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, rank=rank
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, 2 * d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


def test_orthogonal_nsp_conv_fasth_strategy():
    """OrthogonalNSPConv with orth_strategy='fasth' produces valid augmented-state."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
    context_dim = d * stalk_features
    conv = OrthogonalNSPConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, orth_strategy="fasth"
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, 2 * d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()
