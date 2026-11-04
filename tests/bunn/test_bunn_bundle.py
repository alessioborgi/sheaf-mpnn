# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""The flat-bundle structure: per-node orthogonal maps and sync/desync.

Everything BuNN claims rests on the maps really being orthogonal and on
``desync`` really inverting ``sync`` -- that is what makes the transport
:math:`O_v^\\top O_u` a parallel transport and lets Lemma 3.1 factor the
bundle heat kernel through the plain graph one.
"""

import pytest
import torch

from sheaf_mpnn.bunn import BundleMapLearner, SumGNNConv, desync, sync


def ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected ring on ``num_nodes`` nodes (both directions stored)."""
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


NUM_NODES = 6
CONTEXT_DIM = 12


class TestBundleMapLearner:
    @pytest.mark.parametrize("bundle_dim", [2, 3, 4])
    @pytest.mark.parametrize("phi", ["mlp", "sum_gnn"])
    def test_maps_are_orthogonal(self, bundle_dim, phi):
        learner = BundleMapLearner(
            CONTEXT_DIM, num_bundles=3, bundle_dim=bundle_dim, phi=phi
        )
        maps = learner(torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES))

        assert maps.shape == (NUM_NODES, 3, bundle_dim, bundle_dim)
        eye = torch.eye(bundle_dim).expand_as(maps)
        assert torch.allclose(maps @ maps.transpose(-2, -1), eye, atol=1e-5)

    def test_two_dimensional_bundles_split_into_rotations_and_reflections(self):
        """Paper Appendix E: O(2) is disconnected, so half the bundles carry
        det -1 maps -- a rotation-only parameterisation could not reach them.
        """
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=4, bundle_dim=2)
        maps = learner(torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES))

        dets = torch.linalg.det(maps)
        assert torch.allclose(dets[:, :2], torch.ones(NUM_NODES, 2), atol=1e-5)
        assert torch.allclose(dets[:, 2:], -torch.ones(NUM_NODES, 2), atol=1e-5)

    def test_odd_bundle_count_is_allowed(self):
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=3, bundle_dim=2)
        maps = learner(torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES))
        assert maps.shape == (NUM_NODES, 3, 2, 2)

    def test_mlp_phi_ignores_the_graph(self):
        """A node-level MLP must give the same maps whatever the edges are."""
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=2, phi="mlp")
        x = torch.randn(NUM_NODES, CONTEXT_DIM)
        a = learner(x, ring_edge_index(NUM_NODES))
        b = learner(x, torch.empty(2, 0, dtype=torch.long))
        assert torch.allclose(a, b)

    def test_sum_gnn_phi_reads_the_graph(self):
        """The paper's GNN phi must actually depend on connectivity."""
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=2, phi="sum_gnn")
        x = torch.randn(NUM_NODES, CONTEXT_DIM)
        a = learner(x, ring_edge_index(NUM_NODES))
        b = learner(x, torch.empty(2, 0, dtype=torch.long))
        assert not torch.allclose(a, b)

    def test_maps_are_differentiable(self):
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=2)
        maps = learner(torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES))
        maps.sum().backward()
        assert all(p.grad is not None for p in learner.parameters())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"context_dim": 0},
            {"num_bundles": 0},
            {"bundle_dim": 0},
            {"num_layers": 0},
            {"phi": "gcn"},
            {"orth_strategy": "nope"},
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs):
        defaults = {"context_dim": CONTEXT_DIM, "num_bundles": 2}
        defaults.update(kwargs)
        with pytest.raises(ValueError):
            BundleMapLearner(**defaults)  # ty: ignore[invalid-argument-type]


class TestSyncDesync:
    def test_desync_inverts_sync(self):
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=3)
        maps = learner(torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES))
        x = torch.randn(NUM_NODES, 3, 2, 5)
        assert torch.allclose(desync(sync(x, maps), maps), x, atol=1e-5)

    def test_sync_preserves_stalk_norms(self):
        """Orthogonal maps are isometries, which is why the threshold-free
        diffusion cannot inflate a node's signal just by changing frame.
        """
        learner = BundleMapLearner(CONTEXT_DIM, num_bundles=2)
        maps = learner(torch.randn(NUM_NODES, CONTEXT_DIM), ring_edge_index(NUM_NODES))
        x = torch.randn(NUM_NODES, 2, 2, 3)
        assert torch.allclose(sync(x, maps).norm(dim=2), x.norm(dim=2), atol=1e-5)

    def test_sync_applies_the_map_per_node_and_bundle(self):
        maps = torch.stack(
            [torch.eye(2), torch.tensor([[0.0, -1.0], [1.0, 0.0]])]
        ).unsqueeze(0)  # [1, 2, 2, 2]: identity, then a quarter turn.
        x = torch.tensor([[1.0], [0.0]]).expand(2, 1).reshape(1, 1, 2, 1)
        x = x.expand(1, 2, 2, 1).contiguous()

        out = sync(x, maps)
        assert torch.allclose(out[0, 0], x[0, 0])
        assert torch.allclose(out[0, 1], torch.tensor([[0.0], [1.0]]))


class TestSumGNNConv:
    def test_aggregates_neighbours_by_sum(self):
        conv = SumGNNConv(1, 1, bias=False)
        with torch.no_grad():
            conv.linear_self.weight.fill_(0.0)
            conv.linear_ngb.weight.fill_(1.0)

        # Path 0 -- 1 -- 2: node 1 sees both neighbours, the ends see one each.
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        x = torch.tensor([[1.0], [10.0], [100.0]])
        out = conv(x, edge_index)
        assert torch.allclose(out, torch.tensor([[10.0], [101.0], [10.0]]))
