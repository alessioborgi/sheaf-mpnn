# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.nsd.nsd_layers import (
    DiagonalHetNSDConv,
    GeneralHetNSDConv,
    LowRankHetNSDConv,
    OrthogonalHetNSDConv,
)
from tests.testing import set_seed

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


def _reinit_mlp(mlp: nn.Module) -> None:
    """Re-initialize MLP with normal gain so type info propagates visibly."""
    if isinstance(mlp, nn.Sequential):
        for m in mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)


# ---------------------------------------------------------------------------
# Parametrize over layer class x het context flags
# ---------------------------------------------------------------------------

_LAYER_VARIANTS = [
    # (layer_class, extra_kwargs, label)
    (DiagonalHetNSDConv, {}, "diagonal"),
    (GeneralHetNSDConv, {}, "general"),
    (GeneralHetNSDConv, {"use_attention": True}, "general_attention"),
    (OrthogonalHetNSDConv, {}, "orthogonal"),
    (OrthogonalHetNSDConv, {"use_attention": True}, "orthogonal_attention"),
    (OrthogonalHetNSDConv, {"orth_strategy": "fasth"}, "orthogonal_fasth"),
    (LowRankHetNSDConv, {"rank": 2}, "low_rank"),
]

_HET_CONTEXTS = [
    # (use_features, use_node_types, use_edge_types, ensemble, label)
    (True, False, False, False, "features_only"),
    (True, True, False, False, "node_types"),
    (True, False, True, False, "edge_types"),
    (True, True, True, False, "full_type_concat"),
    (False, True, True, False, "types_only"),
    (False, True, False, False, "node_type_only"),
    (False, False, True, False, "edge_type_only"),
    (True, False, False, True, "ensemble"),
    (True, True, False, True, "ensemble_node_types"),
]


def _make_layer(
    layer_class, use_features, use_node_types, use_edge_types, ensemble, extra
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
        **extra,
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
class TestHetNSDVariants:
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
            layer_class, use_features, use_node_types, use_edge_types, ensemble, extra
        )
        return conv, x_feat, x_stalk, edge_index, node_types, edge_types

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index, node_types, edge_types = setup
        out = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()

    def test_isolated_node_stability(self, setup):
        conv, x_feat, x_stalk, _, node_types, _ = setup
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, empty_ei, node_types, None)
        torch.testing.assert_close(out, x_stalk)

    def test_alpha_zero_identity(self, setup):
        conv, x_feat, x_stalk, edge_index, node_types, edge_types = setup
        if not hasattr(conv, "alpha"):
            pytest.skip("no alpha parameter")
        conv.alpha.data.zero_()
        out = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
        torch.testing.assert_close(out, x_stalk)

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index, node_types, edge_types = setup
        out = conv(x_feat, x_stalk, edge_index, node_types, edge_types)
        out.pow(2).sum().backward()
        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        assert conv.W2.grad is not None and conv.W2.grad.abs().sum() > 0

    def test_no_type_fallback(self, setup):
        """Passing None for types when flags are enabled falls back to zeros."""
        conv, x_feat, x_stalk, edge_index, _, _ = setup
        out = conv(x_feat, x_stalk, edge_index, None, None)
        assert out.shape == x_stalk.shape
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Type-awareness: different types -> different output
# ---------------------------------------------------------------------------


def test_type_info_affects_output():
    """The same features but different node_types produce different outputs."""
    set_seed(0)
    context_dim = D * STALK_F
    conv = GeneralHetNSDConv(
        D,
        STALK_F,
        HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=True,
        use_edge_types=False,
        context_dim=context_dim,
    )
    _reinit_mlp(conv.map_generator)
    conv.eval()

    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()

    types_a = torch.zeros(NUM_NODES, dtype=torch.long)
    types_b = torch.ones(NUM_NODES, dtype=torch.long)

    with torch.no_grad():
        out_a = conv(x_feat, x_stalk, edge_index, types_a, None)
        out_b = conv(x_feat, x_stalk, edge_index, types_b, None)

    assert not torch.allclose(out_a, out_b), "Type info should affect output"


def test_edge_type_affects_output():
    """The same features but different edge_types produce different outputs."""
    set_seed(0)
    context_dim = D * STALK_F
    conv = GeneralHetNSDConv(
        D,
        STALK_F,
        HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=False,
        use_edge_types=True,
        context_dim=context_dim,
    )
    _reinit_mlp(conv.map_generator)
    conv.eval()

    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    E = edge_index.size(1)

    types_a = torch.zeros(E, dtype=torch.long)
    types_b = torch.ones(E, dtype=torch.long)

    with torch.no_grad():
        out_a = conv(x_feat, x_stalk, edge_index, None, types_a)
        out_b = conv(x_feat, x_stalk, edge_index, None, types_b)

    assert not torch.allclose(out_a, out_b), "Edge type info should affect output"


# ---------------------------------------------------------------------------
# Ensemble-specific tests
# ---------------------------------------------------------------------------


def test_ensemble_has_per_type_generators():
    """Ensemble mode creates num_edge_types independent MLPs."""
    conv = DiagonalHetNSDConv(
        D,
        STALK_F,
        HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        ensemble=True,
        context_dim=D * STALK_F,
    )
    assert hasattr(conv, "map_generators")
    assert len(conv.map_generators) == NUM_EDGE_TYPES


def test_ensemble_routes_by_type():
    """Ensemble: edges of different types go through different MLPs."""
    set_seed(1)
    context_dim = D * STALK_F
    conv = DiagonalHetNSDConv(
        D,
        STALK_F,
        HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        ensemble=True,
        context_dim=context_dim,
    )
    for gen in conv.map_generators:
        _reinit_mlp(gen)
    conv.eval()

    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    E = edge_index.size(1)

    types_a = torch.zeros(E, dtype=torch.long)
    types_b = torch.ones(E, dtype=torch.long)

    with torch.no_grad():
        out_a = conv(x_feat, x_stalk, edge_index, None, types_a)
        out_b = conv(x_feat, x_stalk, edge_index, None, types_b)

    assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_low_rank_positive_rank_enforced():
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankHetNSDConv(
            D, STALK_F, HIDDEN, rank=0, num_node_types=2, num_edge_types=2
        )


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_various_ranks(rank):
    set_seed(42)
    context_dim = D * STALK_F
    conv = LowRankHetNSDConv(
        D,
        STALK_F,
        HIDDEN,
        rank=rank,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        context_dim=context_dim,
    )
    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Het-context configuration validation (BaseHetSheafConv)
# ---------------------------------------------------------------------------


def test_use_node_types_without_count_raises():
    with pytest.raises(ValueError, match="num_node_types must be > 0"):
        DiagonalHetNSDConv(D, STALK_F, HIDDEN, use_node_types=True, num_node_types=0)


def test_use_edge_types_without_count_raises():
    with pytest.raises(ValueError, match="num_edge_types must be > 0"):
        DiagonalHetNSDConv(D, STALK_F, HIDDEN, use_edge_types=True, num_edge_types=0)


def test_ensemble_without_edge_types_raises():
    with pytest.raises(ValueError, match="num_edge_types must be > 0"):
        DiagonalHetNSDConv(D, STALK_F, HIDDEN, ensemble=True, num_edge_types=0)


def test_all_context_flags_false_raises():
    with pytest.raises(ValueError, match="at least one"):
        DiagonalHetNSDConv(
            D,
            STALK_F,
            HIDDEN,
            use_features=False,
            use_node_types=False,
            use_edge_types=False,
            ensemble=False,
        )


@pytest.mark.parametrize("orth_strategy", ["cayley", "fasth"])
def test_orthogonal_strategies(orth_strategy):
    set_seed(42)
    context_dim = D * STALK_F
    conv = OrthogonalHetNSDConv(
        D,
        STALK_F,
        HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        orth_strategy=orth_strategy,
        context_dim=context_dim,
    )
    x_feat = torch.randn(NUM_NODES, context_dim)
    x_stalk = torch.randn(NUM_NODES, D, STALK_F)
    edge_index = _make_graph()
    out = conv(x_feat, x_stalk, edge_index)
    assert out.shape == x_stalk.shape
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# add_eps: learnable per-stalk skip rescaling (Remark 4)
# ---------------------------------------------------------------------------

_EPS_LAYER_VARIANTS = [
    pytest.param(DiagonalHetNSDConv, {}, id="diagonal"),
    pytest.param(GeneralHetNSDConv, {}, id="general"),
    pytest.param(OrthogonalHetNSDConv, {}, id="orthogonal"),
    pytest.param(LowRankHetNSDConv, {"rank": 2}, id="low_rank"),
]


def _make_eps_layer(layer_class, extra, add_eps):
    context_dim = D * STALK_F
    return layer_class(
        stalk_dim=D,
        in_channels=STALK_F,
        hidden_dim=HIDDEN,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        context_dim=context_dim,
        add_eps=add_eps,
        **extra,
    )


@pytest.mark.parametrize("layer_class,extra", _EPS_LAYER_VARIANTS)
class TestHetNSDAddEps:
    @pytest.fixture
    def inputs(self):
        set_seed(7)
        context_dim = D * STALK_F
        x_feat = torch.randn(NUM_NODES, context_dim)
        x_stalk = torch.randn(NUM_NODES, D, STALK_F)
        edge_index = _make_graph()
        return x_feat, x_stalk, edge_index

    def test_eps_param_only_when_enabled(self, layer_class, extra):
        conv_off = _make_eps_layer(layer_class, extra, add_eps=False)
        assert not hasattr(conv_off, "eps")
        assert conv_off.add_eps is False

        conv_on = _make_eps_layer(layer_class, extra, add_eps=True)
        assert conv_on.add_eps is True
        assert conv_on.eps.shape == (D,)
        torch.testing.assert_close(conv_on.eps, torch.zeros(D))

    def test_zero_eps_matches_disabled(self, layer_class, extra, inputs):
        """Default zero-initialised eps reproduces the add_eps=False update."""
        x_feat, x_stalk, edge_index = inputs

        set_seed(123)
        conv_off = _make_eps_layer(layer_class, extra, add_eps=False)
        set_seed(123)
        conv_on = _make_eps_layer(layer_class, extra, add_eps=True)

        # eval() disables training-only degree-matrix jitter (apply_general_norm),
        # so the comparison isn't sensitive to RNG draws consumed between calls.
        conv_off.eval()
        conv_on.eval()

        with torch.no_grad():
            out_off = conv_off(x_feat, x_stalk, edge_index)
            out_on = conv_on(x_feat, x_stalk, edge_index)

        torch.testing.assert_close(out_off, out_on)

    def test_nonzero_eps_changes_output(self, layer_class, extra, inputs):
        x_feat, x_stalk, edge_index = inputs
        set_seed(123)
        conv = _make_eps_layer(layer_class, extra, add_eps=True)

        with torch.no_grad():
            out_zero = conv(x_feat, x_stalk, edge_index)
            conv.eps.data.fill_(0.5)
            out_shifted = conv(x_feat, x_stalk, edge_index)

        assert not torch.allclose(out_zero, out_shifted)

    def test_eps_receives_gradient(self, layer_class, extra, inputs):
        x_feat, x_stalk, edge_index = inputs
        set_seed(123)
        conv = _make_eps_layer(layer_class, extra, add_eps=True)
        conv.eps.data.fill_(0.3)

        out = conv(x_feat, x_stalk, edge_index)
        out.pow(2).sum().backward()

        assert conv.eps.grad is not None
        assert conv.eps.grad.abs().sum() > 0
