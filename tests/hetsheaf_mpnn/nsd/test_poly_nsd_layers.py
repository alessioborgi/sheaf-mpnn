# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import random

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.nsd.poly_nsd_layers import (
    BasePolyHetNSDConv,
    DiagonalPolyHetNSDConv,
    GeneralPolyHetNSDConv,
    LowRankPolyHetNSDConv,
    OrthogonalPolyHetNSDConv,
)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


NUM_NODES = 10
STALK_F = 8
D = 4
HIDDEN = 8
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _make_graph(num_nodes=NUM_NODES):
    return erdos_renyi_graph(num_nodes, edge_prob=0.4)


def _node_types(num_nodes=NUM_NODES):
    return torch.randint(0, NUM_NODE_TYPES, (num_nodes,))


def _edge_types(edge_index):
    return torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))


# ---------------------------------------------------------------------------
# Parametrize over layer class x het context flags
# ---------------------------------------------------------------------------

_LAYER_VARIANTS = [
    # (layer_class, extra_kwargs, label)
    (DiagonalPolyHetNSDConv, {}, "diagonal"),
    (GeneralPolyHetNSDConv, {}, "general"),
    (GeneralPolyHetNSDConv, {"use_attention": True}, "general_attention"),
    (OrthogonalPolyHetNSDConv, {}, "orthogonal"),
    (OrthogonalPolyHetNSDConv, {"use_attention": True}, "orthogonal_attention"),
    (OrthogonalPolyHetNSDConv, {"orth_strategy": "fasth"}, "orthogonal_fasth"),
    (LowRankPolyHetNSDConv, {"rank": 2}, "low_rank"),
]

_HET_CONTEXTS = [
    # (use_features, use_node_types, use_edge_types, ensemble, label)
    (True, False, False, False, "features_only"),
    (True, True, False, False, "node_types"),
    (True, False, True, False, "edge_types"),
    (True, True, True, False, "full_type_concat"),
    (True, False, False, True, "ensemble"),
]


def _make_layer(
    layer_class,
    use_features=True,
    use_node_types=False,
    use_edge_types=False,
    ensemble=False,
    extra=None,
    **kwargs,
):
    context_dim = D * STALK_F
    return layer_class(
        stalk_dim=D,
        in_channels=STALK_F,
        hidden_dim=HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=use_features,
        use_node_types=use_node_types,
        use_edge_types=use_edge_types,
        ensemble=ensemble,
        context_dim=context_dim,
        **(extra or {}),
        **kwargs,
    )


@pytest.mark.parametrize(
    "layer_class,extra,layer_label",
    [(c, e, lbl) for c, e, lbl in _LAYER_VARIANTS],
    ids=[lbl for _, _, lbl in _LAYER_VARIANTS],
)
@pytest.mark.parametrize(
    "use_features,use_node_types,use_edge_types,ensemble,ctx_label",
    [(f, n, e, ens, lbl) for f, n, e, ens, lbl in _HET_CONTEXTS],
    ids=[lbl for _, _, _, _, lbl in _HET_CONTEXTS],
)
class TestPolyHetNSDVariants:
    """Shared correctness tests for every polynomial het-NSD variant."""

    @pytest.fixture
    def setup(
        self,
        layer_class,
        extra,
        layer_label,
        use_features,
        use_node_types,
        use_edge_types,
        ensemble,
        ctx_label,
    ):
        set_seed(42)
        context_dim = D * STALK_F
        x_feat = torch.randn(NUM_NODES, context_dim)
        x_stalk = torch.randn(NUM_NODES, D, STALK_F)
        edge_index = _make_graph()
        node_types = _node_types()
        edge_types = _edge_types(edge_index)
        conv = _make_layer(
            layer_class,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
            ensemble=ensemble,
            extra=extra,
            degree=2,
        )
        return conv, x_feat, x_stalk, edge_index, node_types, edge_types

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index, node_types, edge_types = setup
        out = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_empty_graph_stability(self, setup):
        """Empty graph (no edges): output must be finite and not nan."""
        conv, x_feat, x_stalk, _, node_types, _ = setup
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, empty_ei, node_types, None)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_no_type_fallback(self, setup):
        """Passing None for types when flags are enabled falls back to zeros."""
        conv, x_feat, x_stalk, edge_index, _, _ = setup
        out = conv(x_feat, x_stalk, edge_index, None, None)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, setup):
        """Gradients must reach W1, W2, theta_raw, hp_alpha, and eps.

        theta_raw is perturbed before the backward pass to break the exact
        cancellation that occurs at the zero initialisation: at theta_raw=0,
        softmax gives equal weights and the Chebyshev recurrence's alternating
        signs can cancel the direct z contribution for even degrees, leaving
        W1/W2 gradients numerically at zero despite the graph being connected.
        """
        conv, x_feat, x_stalk, edge_index, node_types, edge_types = setup
        torch.manual_seed(1)
        conv.theta_raw.data = torch.randn_like(conv.theta_raw)
        out = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
        out.pow(2).sum().backward()

        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        assert conv.W2.grad is not None and conv.W2.grad.abs().sum() > 0
        assert conv.theta_raw.grad is not None and conv.theta_raw.grad.abs().sum() > 0
        assert conv.hp_alpha.grad is not None
        assert conv.eps.grad is not None

    def test_eval_determinism(self, setup):
        """eval() must produce identical outputs on consecutive forward passes."""
        conv, x_feat, x_stalk, edge_index, node_types, edge_types = setup
        conv.eval()
        with torch.no_grad():
            out1 = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
            out2 = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Degree-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_degree_variants_diagonal(degree):
    """DiagonalPolyHetNSDConv must produce valid outputs for degrees 1-5."""
    set_seed(42)
    context_dim = D * STALK_F
    conv = _make_layer(DiagonalPolyHetNSDConv, degree=degree)
    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index, _node_types(), _edge_types(edge_index))
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


def test_degree_zero_raises():
    """Constructing a poly het conv with degree=0 must raise ValueError."""
    with pytest.raises(ValueError, match="degree must be at least 1"):
        _make_layer(DiagonalPolyHetNSDConv, degree=0)


# ---------------------------------------------------------------------------
# Self-loop handling: edge_types must be padded for the added self-loops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_edge_types", [True, False], ids=["types", "no_types"])
def test_add_self_loops_forward(with_edge_types):
    """add_self_loops=True pads edge_types (type 0) for the new self-loop edges."""
    set_seed(42)
    context_dim = D * STALK_F
    conv = _make_layer(
        DiagonalPolyHetNSDConv,
        use_edge_types=with_edge_types,
        add_self_loops=True,
        degree=2,
    )
    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    edge_types = _edge_types(edge_index) if with_edge_types else None
    out = conv(x_feat, x_stalk, edge_index, None, edge_types)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Mixin inheritance and poly-parameter checks
# ---------------------------------------------------------------------------


def test_all_variants_are_poly_instances():
    """All concrete poly het variants must be instances of BasePolyHetNSDConv."""
    convs = [
        _make_layer(DiagonalPolyHetNSDConv),
        _make_layer(GeneralPolyHetNSDConv),
        _make_layer(OrthogonalPolyHetNSDConv),
        _make_layer(LowRankPolyHetNSDConv, extra={"rank": 1}),
    ]
    for conv in convs:
        assert isinstance(conv, BasePolyHetNSDConv)


def test_poly_params_present():
    """Every poly het variant must expose theta_raw, hp_alpha, and eps."""
    conv = _make_layer(DiagonalPolyHetNSDConv, degree=2)
    assert hasattr(conv, "theta_raw") and conv.theta_raw.shape == (3,)
    assert hasattr(conv, "hp_alpha")
    assert hasattr(conv, "eps") and conv.eps.shape == (D,)


def test_theta_raw_softmax_normalised():
    """Softmax of theta_raw must sum to 1."""
    conv = _make_layer(DiagonalPolyHetNSDConv, degree=3)
    theta = torch.softmax(conv.theta_raw, dim=0)
    assert theta.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_eps_gate_init():
    """At initialisation eps=0 so the gate is identically 1."""
    conv = _make_layer(DiagonalPolyHetNSDConv)
    gate = 1.0 + torch.tanh(conv.eps)
    assert gate.allclose(torch.ones_like(gate))


def test_hp_alpha_init_stored():
    """Constructor hp_alpha_init must be reflected in the initial parameter value."""
    conv = _make_layer(DiagonalPolyHetNSDConv, degree=2, hp_alpha_init=0.5)
    assert conv.hp_alpha.item() == pytest.approx(0.5)


def test_reset_parameters_reinitialises_poly_params():
    """reset_parameters must zero out theta_raw, hp_alpha, and eps."""
    set_seed(0)
    conv = _make_layer(DiagonalPolyHetNSDConv, degree=3)
    # Manually corrupt the poly params.
    conv.theta_raw.data.fill_(99.0)
    conv.hp_alpha.data.fill_(99.0)
    conv.eps.data.fill_(99.0)
    conv.reset_parameters()
    assert conv.theta_raw.abs().max().item() == pytest.approx(0.0)
    assert conv.hp_alpha.abs().item() == pytest.approx(0.0)
    assert conv.eps.abs().max().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Orthogonal strategy and low-rank propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orthogonal_poly_strategy_forward(orth_strategy):
    """Both Cayley and FastH strategies must produce valid poly outputs."""
    set_seed(42)
    context_dim = D * STALK_F
    conv = _make_layer(
        OrthogonalPolyHetNSDConv, degree=2, extra={"orth_strategy": orth_strategy}
    )
    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    out = conv(x_feat, x_stalk, edge_index, _node_types(), _edge_types(edge_index))
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_poly_various_ranks(rank):
    """LowRankPolyHetNSDConv must work for ranks 1, 2, 3."""
    set_seed(42)
    context_dim = D * STALK_F
    conv = _make_layer(LowRankPolyHetNSDConv, degree=2, extra={"rank": rank})
    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert torch.isfinite(out).all()
