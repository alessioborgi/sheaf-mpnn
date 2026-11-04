# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.nsp.nsp_layers import (
    DiagonalHetNSPConv,
    GeneralHetNSPConv,
    LowRankHetNSPConv,
    OrthogonalHetNSPConv,
)
from tests.testing import set_seed

NUM_NODES = 10
STALK_F = 8
D = 4
HIDDEN = 8
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _make_graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _make_augmented_stalk():
    return torch.randn(NUM_NODES, 2 * D, STALK_F)


_LAYER_VARIANTS = [
    (DiagonalHetNSPConv, {}, "diagonal"),
    (GeneralHetNSPConv, {}, "general"),
    (GeneralHetNSPConv, {"use_attention": True}, "general_attention"),
    (OrthogonalHetNSPConv, {}, "orthogonal"),
    (OrthogonalHetNSPConv, {"orth_strategy": "fasth"}, "orthogonal_fasth"),
    (LowRankHetNSPConv, {"rank": 2}, "low_rank"),
]

_HET_CONTEXTS = [
    (True, False, False, False, "features_only"),
    (True, True, True, False, "full_type_concat"),
    (True, False, False, True, "ensemble"),
    (True, True, False, True, "ensemble_node_types"),
]


def _make_layer(
    layer_class, use_features, use_node_types, use_edge_types, ensemble, extra
):
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
        context_dim=D * STALK_F,
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
class TestHetNSPVariants:
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
        x_feat = torch.randn(NUM_NODES, D * STALK_F)
        x_stalk = _make_augmented_stalk()
        edge_index = _make_graph()
        node_types = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
        edge_types = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
        conv = _make_layer(
            layer_class, use_features, use_node_types, use_edge_types, ensemble, extra
        )
        return conv, x_feat, x_stalk, edge_index, node_types, edge_types

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index, nt, et = setup
        out = conv(x_feat, x_stalk, edge_index, nt, et)
        assert out.shape == (NUM_NODES, 2 * D, STALK_F)
        assert not torch.isnan(out).any()

    def test_history_update(self, setup):
        """New history block should equal old current block."""
        conv, x_feat, x_stalk, edge_index, nt, et = setup
        out = conv(x_feat, x_stalk, edge_index, nt, et)
        torch.testing.assert_close(
            out[:, D:, :],
            x_stalk[:, :D, :],
            msg="history should be the previous current block",
        )

    def test_isolated_node_recurrence(self, setup):
        """With empty graph, output satisfies wave recurrence exactly."""
        conv, x_feat, x_stalk, _, _, _ = setup
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        x_curr = x_stalk[:, :D, :]
        x_hist = x_stalk[:, D:, :]
        out = conv(x_feat, x_stalk, empty_ei)
        expected_curr = 2 * x_curr - x_hist  # sigma(0) = 0
        torch.testing.assert_close(out[:, :D, :], expected_curr, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index, nt, et = setup
        out = conv(x_feat, x_stalk, edge_index, nt, et)
        out.pow(2).sum().backward()
        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0


def test_low_rank_invalid_rank():
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankHetNSPConv(
            D, STALK_F, HIDDEN, rank=0, num_node_types=2, num_edge_types=2
        )
