# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Type-aware bundle-map learners: the relational SumGNN and het phi."""

import pytest
import torch

from hetsheaf_mpnn.bunn import HetBundleMapLearner, HetSumGNNConv
from sheaf_mpnn.bunn import BundleMapLearner, SumGNNConv
from tests.bunn.test_bunn_bundle import ring_edge_index
from tests.testing import set_seed

NUM_NODES = 6
CONTEXT_DIM = 12
NUM_NODE_TYPES = 2
NUM_EDGE_TYPES = 3


def _edge_types(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.arange(edge_index.size(1)) % NUM_EDGE_TYPES


class TestHetSumGNNConv:
    def test_matches_the_relational_update_edge_by_edge(self):
        set_seed(0)
        conv = HetSumGNNConv(5, 4, NUM_EDGE_TYPES)
        torch.nn.init.normal_(conv.rel_weight)
        edge_index = ring_edge_index(NUM_NODES)
        edge_types = _edge_types(edge_index)
        x = torch.randn(NUM_NODES, 5)

        expected = conv.linear_self(x)
        for (u, v), r in zip(edge_index.t().tolist(), edge_types.tolist(), strict=True):
            expected[v] = (
                expected[v] + conv.linear_ngb(x[u]) + x[u] @ conv.rel_weight[r]
            )
        torch.testing.assert_close(conv(x, edge_index, edge_types), expected)

    def test_starts_as_the_type_blind_sum_gnn(self):
        set_seed(0)
        conv = HetSumGNNConv(5, 4, NUM_EDGE_TYPES)
        base = SumGNNConv(5, 4)
        base.load_state_dict(conv.state_dict(), strict=False)
        edge_index = ring_edge_index(NUM_NODES)
        x = torch.randn(NUM_NODES, 5)
        torch.testing.assert_close(
            conv(x, edge_index, _edge_types(edge_index)), base(x, edge_index)
        )

    def test_reset_zeroes_the_relation_weights(self):
        conv = HetSumGNNConv(5, 4, NUM_EDGE_TYPES)
        torch.nn.init.normal_(conv.rel_weight)
        conv.reset_parameters()
        assert torch.count_nonzero(conv.rel_weight) == 0

    def test_without_edge_types_skips_the_relation_term(self):
        conv = HetSumGNNConv(5, 4, NUM_EDGE_TYPES)
        torch.nn.init.normal_(conv.rel_weight)
        edge_index = ring_edge_index(NUM_NODES)
        x = torch.randn(NUM_NODES, 5)
        expected = SumGNNConv.forward(conv, x, edge_index)
        torch.testing.assert_close(conv(x, edge_index), expected)

    def test_invalid_num_edge_types(self):
        with pytest.raises(ValueError, match="num_edge_types must be positive"):
            HetSumGNNConv(5, 4, 0)


class TestHetBundleMapLearner:
    @pytest.mark.parametrize("bundle_dim", [2, 3])
    @pytest.mark.parametrize("phi", ["mlp", "sum_gnn"])
    def test_maps_are_orthogonal(self, bundle_dim, phi):
        learner = HetBundleMapLearner(
            CONTEXT_DIM,
            num_bundles=3,
            bundle_dim=bundle_dim,
            phi=phi,
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            use_node_types=True,
            use_edge_types=phi == "sum_gnn",
        )
        edge_index = ring_edge_index(NUM_NODES)
        node_types = torch.arange(NUM_NODES) % NUM_NODE_TYPES
        maps = learner(
            torch.randn(NUM_NODES, CONTEXT_DIM),
            edge_index,
            node_types,
            _edge_types(edge_index),
        )
        assert maps.shape == (NUM_NODES, 3, bundle_dim, bundle_dim)
        eye = torch.eye(bundle_dim).expand_as(maps)
        torch.testing.assert_close(
            maps @ maps.transpose(-1, -2), eye, atol=1e-5, rtol=0
        )

    @pytest.mark.parametrize("phi", ["mlp", "sum_gnn"])
    def test_features_only_matches_the_homogeneous_learner(self, phi):
        set_seed(0)
        het = HetBundleMapLearner(CONTEXT_DIM, num_bundles=3, phi=phi)
        base = BundleMapLearner(CONTEXT_DIM, num_bundles=3, phi=phi)
        base.load_state_dict(het.state_dict())
        x = torch.randn(NUM_NODES, CONTEXT_DIM)
        edge_index = ring_edge_index(NUM_NODES)
        torch.testing.assert_close(het(x, edge_index), base(x, edge_index))

    def test_type_only_maps_give_one_frame_per_node_type(self):
        # With no features, an MLP phi sees only the type: same type, same frame.
        set_seed(0)
        learner = HetBundleMapLearner(
            CONTEXT_DIM,
            num_bundles=3,
            use_features=False,
            num_node_types=NUM_NODE_TYPES,
            use_node_types=True,
        )
        node_types = torch.arange(NUM_NODES) % NUM_NODE_TYPES
        maps = learner(
            torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES), node_types
        )
        for t in range(NUM_NODE_TYPES):
            same = maps[node_types == t]
            torch.testing.assert_close(same, same[:1].expand_as(same))
        assert not torch.allclose(maps[0], maps[1])

    def test_edge_types_reach_the_maps(self):
        set_seed(0)
        learner = HetBundleMapLearner(
            CONTEXT_DIM,
            num_bundles=3,
            phi="sum_gnn",
            num_edge_types=NUM_EDGE_TYPES,
            use_edge_types=True,
        )
        for conv in learner.convs:
            assert isinstance(conv, HetSumGNNConv)
            torch.nn.init.normal_(conv.rel_weight)
        x = torch.randn(NUM_NODES, CONTEXT_DIM)
        edge_index = ring_edge_index(NUM_NODES)
        edge_types = _edge_types(edge_index)
        assert not torch.allclose(
            learner(x, edge_index, edge_types=edge_types),
            learner(x, edge_index, edge_types=(edge_types + 1) % NUM_EDGE_TYPES),
        )

    def test_missing_node_types_fall_back_to_zero_one_hots(self):
        learner = HetBundleMapLearner(
            CONTEXT_DIM,
            num_bundles=3,
            num_node_types=NUM_NODE_TYPES,
            use_node_types=True,
        )
        assert isinstance(learner.mlp[0], torch.nn.Linear)
        torch.nn.init.zeros_(learner.mlp[0].weight[:, CONTEXT_DIM:])
        x = torch.randn(NUM_NODES, CONTEXT_DIM)
        edge_index = ring_edge_index(NUM_NODES)
        node_types = torch.arange(NUM_NODES) % NUM_NODE_TYPES
        # With the type columns zeroed, passing types or not is the same input.
        torch.testing.assert_close(
            learner(x, edge_index), learner(x, edge_index, node_types)
        )

    def test_pe_and_node_types_compose(self):
        learner = HetBundleMapLearner(
            CONTEXT_DIM,
            num_bundles=3,
            pe_size=4,
            num_node_types=NUM_NODE_TYPES,
            use_node_types=True,
        )
        assert isinstance(learner.mlp[0], torch.nn.Linear)
        assert learner.mlp[0].in_features == CONTEXT_DIM + 4 + NUM_NODE_TYPES
        maps = learner(
            torch.randn(NUM_NODES, CONTEXT_DIM),
            ring_edge_index(NUM_NODES),
            torch.zeros(NUM_NODES, dtype=torch.long),
            pe=torch.randn(NUM_NODES, 4),
        )
        assert maps.shape == (NUM_NODES, 3, 2, 2)

    def test_pe_is_required_and_checked(self):
        learner = HetBundleMapLearner(CONTEXT_DIM, num_bundles=3, pe_size=4)
        x = torch.randn(NUM_NODES, CONTEXT_DIM)
        edge_index = ring_edge_index(NUM_NODES)
        with pytest.raises(ValueError, match="pass pe to forward"):
            learner(x, edge_index)
        with pytest.raises(ValueError, match="pe has width"):
            learner(x, edge_index, pe=torch.randn(NUM_NODES, 3))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"use_edge_types": True, "num_edge_types": 3}, "requires phi='sum_gnn'"),
            ({"use_node_types": True}, "num_node_types must be > 0"),
            (
                {"use_edge_types": True, "phi": "sum_gnn"},
                "num_edge_types must be > 0",
            ),
            ({"use_features": False}, "phi has no input"),
            ({"pe_size": -1}, "pe_size must be non-negative"),
        ],
    )
    def test_invalid_configurations_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HetBundleMapLearner(CONTEXT_DIM, num_bundles=3, **kwargs)
