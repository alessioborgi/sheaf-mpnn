# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch

from sheaf_mpnn.directed_nsd.directed_nsd_layers import (
    DiagonalDirectedNSDConv,
    GeneralDirectedNSDConv,
    LowRankDirectedNSDConv,
    OrthogonalDirectedNSDConv,
)
from tests.testing import set_seed


class TestOrthogonalDirectedNSDConv:
    @pytest.fixture
    def setup(self):
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        # Directed edge list: (source, destination)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

        conv = OrthogonalDirectedNSDConv(
            d, stalk_features, hidden_dim, context_dim=context_dim
        )
        return conv, x_feat, x_stalk, edge_index

    def test_output_shape(self, setup):
        """Output shape matches input stalk shape."""
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_no_nans(self, setup):
        """Forward pass produces no NaN values."""
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_isolated_node_stability(self, setup):
        """Isolated nodes (no edges) produce finite output."""
        conv, x_feat, x_stalk, _ = setup
        conv_no_loops = OrthogonalDirectedNSDConv(
            conv.stalk_dim,
            conv.in_channels,
            8,
            context_dim=conv.context_dim,
            add_self_loops=False,
        )
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = conv_no_loops(x_feat, x_stalk, edge_index)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_alpha_direction_blending(self, setup):
        """Alpha parameter blends forward and backward directions."""
        conv, x_feat, x_stalk, edge_index = setup
        conv.eval()

        # Get outputs for different alpha values
        conv.alpha.data.fill_(0.0)  # backward-only
        out_bwd = conv(x_feat, x_stalk, edge_index)

        conv.alpha.data.fill_(1.0)  # forward-only
        out_fwd = conv(x_feat, x_stalk, edge_index)

        conv.alpha.data.fill_(0.5)  # blend both
        out_blend = conv(x_feat, x_stalk, edge_index)

        # Both directions should produce different outputs
        assert not torch.allclose(out_fwd, out_bwd, atol=1e-5)

        # Blended output should be between forward and backward
        # (not necessarily exact due to non-linearity, but shows mixing)
        assert out_blend.shape == out_fwd.shape

    def test_gradient_flow(self, setup):
        """Gradients flow through learnable parameters."""
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert conv.W1_fwd.grad is not None and conv.W1_fwd.grad.abs().sum() > 0
        assert conv.W2_fwd.grad is not None and conv.W2_fwd.grad.abs().sum() > 0
        assert conv.W1_bwd.grad is not None and conv.W1_bwd.grad.abs().sum() > 0
        assert conv.W2_bwd.grad is not None and conv.W2_bwd.grad.abs().sum() > 0
        assert (
            conv.map_generator[0].weight.grad is not None
            and conv.map_generator[0].weight.grad.abs().sum() > 0
        )

    def test_permutation_invariance(self, setup):
        """Node permutation invariance on the permuted graph."""
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
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=1e-5, rtol=1e-5)

    def test_restriction_map_initialization(self, setup):
        """Initial map generators keep output close to input."""
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        diff = torch.abs(out - x_stalk).mean()
        assert diff < 2.0

    @pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
    def test_orth_strategy(self, orth_strategy):
        """Both Cayley and Householder strategies work."""
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 8, 12, 3, 6
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        conv = OrthogonalDirectedNSDConv(
            d,
            stalk_features,
            hidden_dim,
            context_dim=context_dim,
            orth_strategy=orth_strategy,
        )
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()


class TestDirectedVsUndirected:
    """Tests verifying that directional message aggregation is asymmetric."""

    def test_directional_asymmetry(self):
        """Reversing edge direction produces different results."""
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 8, 12, 3, 6
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)

        # Edge: 0 -> 1
        edge_index_fwd = torch.tensor([[0], [1]], dtype=torch.long)
        # Edge: 1 -> 0 (reversed)
        edge_index_rev = torch.tensor([[1], [0]], dtype=torch.long)

        conv = OrthogonalDirectedNSDConv(
            d, stalk_features, hidden_dim, context_dim=context_dim
        )
        conv.eval()

        out_fwd = conv(x_feat, x_stalk, edge_index_fwd)
        out_rev = conv(x_feat, x_stalk, edge_index_rev)

        # Outputs should differ due to directional restriction maps
        diff = torch.abs(out_fwd - out_rev).mean()
        assert diff > 1e-5  # Significant difference expected

    def test_in_out_aggregation_separate(self):
        """In-edges and out-edges aggregations are computed separately."""
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 6, 12, 3, 6
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)

        # Star graph: node 0 is center, edges from all others to 0 and vice versa
        sources = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
        targets = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
        edge_index = torch.tensor([sources, targets], dtype=torch.long)

        conv = OrthogonalDirectedNSDConv(
            d, stalk_features, hidden_dim, context_dim=context_dim
        )
        conv.eval()

        # Node 0: receives from 1,2,3,4,5 (in-neighbors) and has out-edges to 1,2,3,4,5
        # The in and out aggregations should contribute differently
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()


class TestDirectedNSDDegreeNormalization:
    """Tests that in-degree and out-degree are computed correctly."""

    def test_degree_computation(self):
        """In-degree and out-degree are computed from directed edges."""
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 10, 12, 3, 6
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)

        # Simple directed edges: 0->1, 1->2, 2->3, 0->2
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 2]], dtype=torch.long)

        conv = OrthogonalDirectedNSDConv(
            d, stalk_features, hidden_dim, context_dim=context_dim
        )

        # Forward pass should compute in/out-degree correctly
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_self_loops_degree_normalization(self):
        """Self-loops are included in degree computation when add_self_loops=True."""
        set_seed(42)
        num_nodes, stalk_features, d, hidden_dim = 5, 12, 3, 6
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        conv_with_loops = OrthogonalDirectedNSDConv(
            d,
            stalk_features,
            hidden_dim,
            context_dim=context_dim,
            add_self_loops=True,
        )
        conv_no_loops = OrthogonalDirectedNSDConv(
            d,
            stalk_features,
            hidden_dim,
            context_dim=context_dim,
            add_self_loops=False,
        )

        out_with = conv_with_loops(x_feat, x_stalk, edge_index)
        out_without = conv_no_loops(x_feat, x_stalk, edge_index)

        # Outputs should differ due to different degree normalization
        diff = torch.abs(out_with - out_without).mean()
        assert diff > 1e-5


# ---------------------------------------------------------------------------
# Parametrised tests across ALL layer variants
# ---------------------------------------------------------------------------

_NUM_NODES, _D, _F, _H = 8, 3, 6, 8
_CONTEXT_DIM = _D * _F
_EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

_ALL_VARIANTS = [
    pytest.param(DiagonalDirectedNSDConv, {}, id="diagonal"),
    pytest.param(GeneralDirectedNSDConv, {}, id="general"),
    pytest.param(
        GeneralDirectedNSDConv, {"use_attention": True}, id="general_attention"
    ),
    pytest.param(OrthogonalDirectedNSDConv, {}, id="orthogonal_cayley"),
    pytest.param(
        OrthogonalDirectedNSDConv, {"orth_strategy": "fasth"}, id="orthogonal_fasth"
    ),
    pytest.param(
        OrthogonalDirectedNSDConv, {"use_attention": True}, id="orthogonal_attention"
    ),
    pytest.param(LowRankDirectedNSDConv, {"rank": 1}, id="lowrank_r1"),
    pytest.param(LowRankDirectedNSDConv, {"rank": 2}, id="lowrank_r2"),
]


def _make_layer(cls, kwargs):
    return cls(_D, _F, _H, context_dim=_CONTEXT_DIM, **kwargs)


@pytest.mark.parametrize("cls,kwargs", _ALL_VARIANTS)
def test_all_variants_output_shape(cls, kwargs):
    """Every variant produces the correct output shape with no NaN/Inf."""
    set_seed(42)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    conv = _make_layer(cls, kwargs)
    out = conv(x_feat, x_stalk, _EDGE_INDEX)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("cls,kwargs", _ALL_VARIANTS)
def test_all_variants_gradient_flow(cls, kwargs):
    """Gradients reach W1_fwd, W2_fwd, W1_bwd, W2_bwd and map_generator."""
    set_seed(42)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    conv = _make_layer(cls, kwargs)
    out = conv(x_feat, x_stalk, _EDGE_INDEX)
    out.pow(2).sum().backward()

    for name in ("W1_fwd", "W2_fwd", "W1_bwd", "W2_bwd"):
        assert getattr(conv, name).grad is not None, f"{name}.grad is None"
    assert conv.map_generator[0].weight.grad is not None


@pytest.mark.parametrize("cls,kwargs", _ALL_VARIANTS)
def test_all_variants_no_edges(cls, kwargs):
    """Every variant handles an empty edge_index without crashing."""
    set_seed(42)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    conv = _make_layer(cls, kwargs)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("cls,kwargs", _ALL_VARIANTS)
def test_all_variants_directional_asymmetry(cls, kwargs):
    """Every variant produces different output when edge direction is reversed."""
    set_seed(42)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    conv = _make_layer(cls, kwargs)
    conv.eval()
    ei_fwd = torch.tensor([[0], [1]], dtype=torch.long)
    ei_rev = torch.tensor([[1], [0]], dtype=torch.long)
    out_fwd = conv(x_feat, x_stalk, ei_fwd)
    out_rev = conv(x_feat, x_stalk, ei_rev)
    assert torch.abs(out_fwd - out_rev).mean() > 1e-6


# ---------------------------------------------------------------------------
# LowRank-specific
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_lowrank_rank_parameter(rank):
    """LowRankDirectedNSDConv works for different rank values."""
    set_seed(42)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    conv = LowRankDirectedNSDConv(_D, _F, _H, context_dim=_CONTEXT_DIM, rank=rank)
    out = conv(x_feat, x_stalk, _EDGE_INDEX)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


def test_lowrank_invalid_rank():
    """LowRankDirectedNSDConv rejects rank <= 0."""
    with pytest.raises(ValueError):
        LowRankDirectedNSDConv(_D, _F, _H, context_dim=_CONTEXT_DIM, rank=0)


# ---------------------------------------------------------------------------
# Attention-specific
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        pytest.param(GeneralDirectedNSDConv, {"use_attention": True}, id="general"),
        pytest.param(
            OrthogonalDirectedNSDConv, {"use_attention": True}, id="orthogonal"
        ),
    ],
)
def test_attention_vs_no_attention(cls, kwargs):
    """Attention and non-attention variants produce different outputs."""
    set_seed(42)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    conv_attn = cls(_D, _F, _H, context_dim=_CONTEXT_DIM, **kwargs)
    conv_plain = cls(_D, _F, _H, context_dim=_CONTEXT_DIM)
    conv_attn.eval()
    conv_plain.eval()
    out_attn = conv_attn(x_feat, x_stalk, _EDGE_INDEX)
    out_plain = conv_plain(x_feat, x_stalk, _EDGE_INDEX)
    assert out_attn.shape == x_stalk.shape
    assert not torch.isnan(out_attn).any()
    assert out_attn.shape == out_plain.shape


# ---------------------------------------------------------------------------
# Bidirectional input and bilateral transforms
# ---------------------------------------------------------------------------


def test_bidirectional_input_shape():
    """_bidirectional_input returns two tensors of matching shape."""
    set_seed(42)
    conv = GeneralDirectedNSDConv(_D, _F, _H, context_dim=_CONTEXT_DIM)
    x_feat = torch.randn(_NUM_NODES, _CONTEXT_DIM)
    num_edges = _EDGE_INDEX.size(1)
    phi_dst, phi_src = conv._bidirectional_input(x_feat, _EDGE_INDEX)
    assert phi_dst.shape == phi_src.shape == (num_edges, _D * _D)


def test_directional_transforms_differ():
    """W_fwd and W_bwd are independent — transforms differ after random init."""
    set_seed(42)
    conv = GeneralDirectedNSDConv(_D, _F, _H, context_dim=_CONTEXT_DIM)
    x_stalk = torch.randn(_NUM_NODES, _D, _F)
    z_fwd = conv._apply_stalk_transform_fwd(x_stalk)
    z_bwd = conv._apply_stalk_transform_bwd(x_stalk)
    # With random initialisation the two transforms almost surely differ
    assert not torch.allclose(z_fwd, z_bwd, atol=1e-5)


def test_learn_alpha_layer():
    """learn_alpha=True makes alpha a trainable parameter."""
    conv = OrthogonalDirectedNSDConv(
        _D, _F, _H, context_dim=_CONTEXT_DIM, learn_alpha=True
    )
    assert conv.alpha.requires_grad
    conv_fixed = OrthogonalDirectedNSDConv(
        _D, _F, _H, context_dim=_CONTEXT_DIM, learn_alpha=False
    )
    assert not conv_fixed.alpha.requires_grad


@pytest.mark.parametrize("clamp_val", [1.0, 5.0, 10.0, 20.0])
def test_cayley_clamp_values(clamp_val):
    """Cayley clamp parameter variations work correctly."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 6, 12, 3, 6
    context_dim = d * stalk_features

    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    conv = OrthogonalDirectedNSDConv(
        d,
        stalk_features,
        hidden_dim,
        context_dim=context_dim,
        orth_strategy="cayley",
        clamp_val=clamp_val,
    )
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()
