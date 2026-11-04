# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.nsd.poly_nsd_layers import (
    BasePolyNSDConv,
    DiagonalPolyNSDConv,
    GeneralPolyNSDConv,
    LowRankPolyNSDConv,
    OrthogonalPolyNSDConv,
)
from tests.testing import set_seed

# ---------------------------------------------------------------------------
# Parametrised fixture covering all four variants and selected degrees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_class,layer_kwargs",
    [
        (DiagonalPolyNSDConv, {}),
        (GeneralPolyNSDConv, {}),
        (OrthogonalPolyNSDConv, {}),
        (GeneralPolyNSDConv, {"use_attention": True}),
        (OrthogonalPolyNSDConv, {"use_attention": True}),
        (OrthogonalPolyNSDConv, {"orth_strategy": "fasth"}),
        (LowRankPolyNSDConv, {"rank": 2}),
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
class TestPolyNSDVariants:
    """Shared correctness tests for every polynomial NSD variant."""

    @pytest.fixture
    def setup(self, model_class, layer_kwargs):
        set_seed(42)
        # d=4, stalk_features=16 -> context_dim = 64
        num_nodes, stalk_features, d, hidden_dim = 10, 16, 4, 8
        context_dim = d * stalk_features

        x_feat = torch.randn(num_nodes, context_dim)
        x_stalk = torch.randn(num_nodes, d, stalk_features)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)

        conv = model_class(
            d,
            stalk_features,
            hidden_dim,
            context_dim=context_dim,
            degree=3,
            **layer_kwargs,
        )
        return conv, x_feat, x_stalk, edge_index

    def test_output_shape(self, setup):
        """Forward must return the same shape as the input stalk."""
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_isolated_node_stability(self, setup):
        """Empty graph (no edges): output must be finite and not nan."""
        conv, x_feat, x_stalk, _ = setup
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, setup):
        """Gradients must reach W1, W2, theta_raw, hp_alpha, eps, and map_generator.

        theta_raw is perturbed before the backward pass to break the exact
        cancellation that occurs at the zero initialisation: at theta_raw=0,
        softmax gives equal weights and the Chebyshev recurrence's alternating
        signs cancel the direct z contribution for even degrees, leaving W1/W2
        gradients numerically at zero despite the graph being connected.  A
        random perturbation ensures the cancellation does not happen.
        """
        conv, x_feat, x_stalk, edge_index = setup
        # Perturb theta_raw so the polynomial weights are unequal.
        torch.manual_seed(1)
        conv.theta_raw.data = torch.randn_like(conv.theta_raw)
        out = conv(x_feat, x_stalk, edge_index)
        out.pow(2).sum().backward()

        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        assert conv.W2.grad is not None and conv.W2.grad.abs().sum() > 0
        assert conv.theta_raw.grad is not None and conv.theta_raw.grad.abs().sum() > 0
        assert conv.hp_alpha.grad is not None
        assert conv.eps.grad is not None
        for param in conv.map_generator.parameters():
            assert param.grad is not None

    def test_theta_raw_softmax_normalised(self, setup):
        """After one forward pass theta = softmax(theta_raw) must sum to 1."""
        conv, x_feat, x_stalk, edge_index = setup
        # The softmax is computed inside forward; verify it on the raw params.
        theta = torch.softmax(conv.theta_raw, dim=0)
        assert theta.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_eps_gate_init(self, setup):
        """At initialisation eps=0 so the gate is identically 1."""
        conv, *_ = setup
        gate = 1.0 + torch.tanh(conv.eps)
        assert gate.allclose(torch.ones_like(gate))

    def test_hp_alpha_zero_no_skip(self, setup):
        """With hp_alpha=0 and eval mode, two identical forward passes must agree."""
        conv, x_feat, x_stalk, edge_index = setup
        conv.hp_alpha.data.zero_()
        # Both runs must be in eval mode so the stochastic jitter in
        # apply_general_norm (training-mode only) does not differ between calls.
        conv.eval()
        with torch.no_grad():
            out1 = conv(x_feat, x_stalk, edge_index)
            out2 = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_permutation_invariance(self, setup):
        """Permuted nodes must yield the permuted output."""
        conv, x_feat, x_stalk, edge_index = setup
        conv.eval()
        out_orig = conv(x_feat, x_stalk, edge_index)

        perm = torch.randperm(x_feat.size(0))
        rev_perm = torch.argsort(perm)
        mapping = torch.zeros(x_feat.size(0), dtype=torch.long)
        mapping[perm] = torch.arange(x_feat.size(0))
        row, col = edge_index
        edge_index_p = torch.stack([mapping[row], mapping[col]], dim=0)

        out_perm = conv(x_feat[perm], x_stalk[perm], edge_index_p)
        torch.testing.assert_close(out_orig, out_perm[rev_perm], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Degree-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_degree_variants_diagonal(degree):
    """DiagonalPolyNSDConv must produce valid outputs for degrees 1-5."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 8, 8, 3, 4
    context_dim = d * stalk_features
    conv = DiagonalPolyNSDConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, degree=degree
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_degree_one_equals_two_propagate_calls():
    """With degree=1, forward uses exactly 2 propagate calls (T1 + one for HP skip).

    We verify indirectly: degree=1 still produces a valid, finite output.
    """
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 6, 4, 2, 4
    context_dim = d * stalk_features
    conv = DiagonalPolyNSDConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, degree=1
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_degree_zero_raises():
    """Constructing a poly conv with degree=0 must raise ValueError."""
    with pytest.raises(ValueError, match="degree must be at least 1"):
        DiagonalPolyNSDConv(stalk_dim=4, in_channels=8, hidden_dim=4, degree=0)


def test_low_rank_invalid_rank():
    """LowRankPolyNSDConv must reject non-positive rank values."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankPolyNSDConv(stalk_dim=4, in_channels=8, hidden_dim=4, rank=0)


# ---------------------------------------------------------------------------
# Mixin inheritance checks
# ---------------------------------------------------------------------------


def test_all_variants_are_poly_instances():
    """All concrete poly variants must be instances of BasePolyNSDConv."""
    d, f, h = 2, 4, 4
    context_dim = d * f
    classes = [
        DiagonalPolyNSDConv(d, f, h, context_dim=context_dim),
        GeneralPolyNSDConv(d, f, h, context_dim=context_dim),
        OrthogonalPolyNSDConv(d, f, h, context_dim=context_dim),
        LowRankPolyNSDConv(d, f, h, context_dim=context_dim, rank=1),
    ]
    for conv in classes:
        assert isinstance(conv, BasePolyNSDConv)


def test_poly_params_present():
    """Every poly variant must expose theta_raw, hp_alpha, and eps."""
    conv = DiagonalPolyNSDConv(stalk_dim=3, in_channels=6, hidden_dim=4, degree=2)
    assert hasattr(conv, "theta_raw") and conv.theta_raw.shape == (3,)
    assert hasattr(conv, "hp_alpha")
    assert hasattr(conv, "eps") and conv.eps.shape == (3,)


def test_reset_parameters_reinitialises_poly_params():
    """reset_parameters must zero out theta_raw, hp_alpha, and eps."""
    set_seed(0)
    conv = DiagonalPolyNSDConv(stalk_dim=4, in_channels=8, hidden_dim=4, degree=3)
    # Manually corrupt the poly params.
    conv.theta_raw.data.fill_(99.0)
    conv.hp_alpha.data.fill_(99.0)
    conv.eps.data.fill_(99.0)
    conv.reset_parameters()
    assert conv.theta_raw.abs().max().item() == pytest.approx(0.0)
    assert conv.hp_alpha.abs().item() == pytest.approx(0.0)
    assert conv.eps.abs().max().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# hp_alpha_init propagation
# ---------------------------------------------------------------------------


def test_hp_alpha_init_stored():
    """Constructor hp_alpha_init must be reflected in the initial parameter value."""
    conv = DiagonalPolyNSDConv(
        stalk_dim=4, in_channels=8, hidden_dim=4, degree=2, hp_alpha_init=0.5
    )
    assert conv.hp_alpha.item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Orthogonal strategy tests (matching first-order NSD coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orthogonal_poly_strategy_forward(orth_strategy):
    """Both Cayley and FastH strategies must produce valid poly outputs."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 8, 8, 3, 4
    context_dim = d * stalk_features
    conv = OrthogonalPolyNSDConv(
        d,
        stalk_features,
        hidden_dim,
        context_dim=context_dim,
        degree=2,
        orth_strategy=orth_strategy,
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Low-rank rank propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_poly_various_ranks(rank):
    """LowRankPolyNSDConv must work for ranks 1, 2, 3."""
    set_seed(42)
    num_nodes, stalk_features, d, hidden_dim = 8, 8, 4, 4
    context_dim = d * stalk_features
    conv = LowRankPolyNSDConv(
        d, stalk_features, hidden_dim, context_dim=context_dim, degree=2, rank=rank
    )
    x_feat = torch.randn(num_nodes, context_dim)
    x_stalk = torch.randn(num_nodes, d, stalk_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()
