# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Layer-level tests for the heterogeneous NLSD convolutions.

The diffusion operator is the homogeneous one; what is new here is that the
restriction maps come from the het context. These tests therefore concentrate
on the het-specific surface -- context flags, the ensemble routing, type
tensors being optional -- plus the invariants the homogeneous layer guarantees
and which the het rewrite must not break (permutation equivariance, an empty
graph reducing to a pure skip, self-loops staying inert).
"""

import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph, to_undirected

from hetsheaf_mpnn.nlsd.nlsd_layers import (
    DiagonalHetNLSDConv,
    GeneralHetNLSDConv,
    LowRankHetNLSDConv,
    OrthogonalHetNLSDConv,
)
from sheaf_mpnn.nlsd.nonlinearity import (
    IdentityNonlinearity,
    MLPNonlinearity,
    ScalarThresholdNonlinearity,
)

NUM_NODES = 12
STALK_DIM = 3
CHANNELS = 4
HIDDEN_DIM = 8
CONTEXT_DIM = STALK_DIM * CHANNELS
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4

LAYER_CLASSES = [
    DiagonalHetNLSDConv,
    OrthogonalHetNLSDConv,
    GeneralHetNLSDConv,
    LowRankHetNLSDConv,
]


def make_graph(seed: int = 0, edge_prob: float = 0.4):
    torch.manual_seed(seed)
    x_feat = torch.randn(NUM_NODES, CONTEXT_DIM)
    x_stalk = torch.randn(NUM_NODES, STALK_DIM, CHANNELS)
    edge_index = to_undirected(erdos_renyi_graph(NUM_NODES, edge_prob=edge_prob))
    node_types = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    edge_types = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return x_feat, x_stalk, edge_index, node_types, edge_types


def make_conv(cls, phi=None, **kwargs):
    kwargs.setdefault("num_node_types", NUM_NODE_TYPES)
    kwargs.setdefault("num_edge_types", NUM_EDGE_TYPES)
    kwargs.setdefault("use_node_types", True)
    kwargs.setdefault("use_edge_types", True)
    return cls(
        STALK_DIM,
        CHANNELS,
        HIDDEN_DIM,
        nonlinearity=phi if phi is not None else MLPNonlinearity(STALK_DIM, CHANNELS),
        context_dim=CONTEXT_DIM,
        **kwargs,
    )


@pytest.mark.parametrize("cls", LAYER_CLASSES, ids=lambda c: c.__name__)
class TestHetNLSDConvVariants:
    def test_output_shape(self, cls):
        x_feat, x_stalk, edge_index, nt, et = make_graph()
        out = make_conv(cls)(x_feat, x_stalk, edge_index, nt, et)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, cls):
        x_feat, x_stalk, edge_index, nt, et = make_graph()
        conv = make_conv(cls)
        conv(x_feat, x_stalk, edge_index, nt, et).pow(2).sum().backward()
        grads = [p.grad for p in conv.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_types_are_optional(self, cls):
        """A het layer must still run when no type tensors are supplied."""
        x_feat, x_stalk, edge_index, _, _ = make_graph()
        out = make_conv(cls)(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_empty_graph_is_a_pure_skip(self, cls):
        """With no edges the diffusion term vanishes and only the skip remains."""
        x_feat, x_stalk, _, nt, _ = make_graph()
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_types = torch.empty(0, dtype=torch.long)
        conv = make_conv(cls, add_eps=False, per_stalk_alpha=False)
        conv.eval()
        with torch.no_grad():
            out = conv(x_feat, x_stalk, edge_index, nt, edge_types)
        torch.testing.assert_close(out, x_stalk)

    def test_ensemble_routes_through_per_edge_type_generators(self, cls):
        x_feat, x_stalk, edge_index, nt, et = make_graph()
        conv = make_conv(cls, ensemble=True, use_edge_types=False)
        assert len(conv.map_generators) == NUM_EDGE_TYPES
        out = conv(x_feat, x_stalk, edge_index, nt, et)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_permutation_equivariance(self, cls):
        """Relabelling the nodes permutes the output and nothing else."""
        x_feat, x_stalk, edge_index, nt, et = make_graph(seed=5)
        conv = make_conv(cls)
        conv.eval()
        perm = torch.randperm(NUM_NODES)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(NUM_NODES)

        with torch.no_grad():
            out = conv(x_feat, x_stalk, edge_index, nt, et)
            out_perm = conv(x_feat[perm], x_stalk[perm], inv[edge_index], nt[perm], et)
        torch.testing.assert_close(out_perm, out[perm], atol=1e-5, rtol=1e-4)


class TestHetContextReachesTheMaps:
    """The whole point of the het family: type information must matter.

    These assert on the restriction maps rather than the layer output. The het
    generators are warm-started near zero (``gain=0.01`` in
    ``BaseHetSheafConv.reset_parameters``, so the Laplacian is off at init),
    which makes the type-driven difference real but far below the output's
    skip term at initialisation.
    """

    def test_node_types_change_the_maps(self):
        x_feat, _, edge_index, nt, et = make_graph()
        conv = make_conv(GeneralHetNLSDConv, use_edge_types=False)
        conv.eval()
        with torch.no_grad():
            a, _ = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES, nt, et)
            b, _ = conv.get_restriction_maps(
                x_feat, edge_index, NUM_NODES, (nt + 1) % NUM_NODE_TYPES, et
            )
        assert not torch.allclose(a, b)

    def test_edge_types_change_the_maps(self):
        x_feat, _, edge_index, nt, et = make_graph()
        conv = make_conv(GeneralHetNLSDConv, use_node_types=False)
        conv.eval()
        with torch.no_grad():
            a, _ = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES, nt, et)
            b, _ = conv.get_restriction_maps(
                x_feat, edge_index, NUM_NODES, nt, (et + 1) % NUM_EDGE_TYPES
            )
        assert not torch.allclose(a, b)

    def test_ensemble_routing_changes_the_maps(self):
        """In ensemble mode the edge type selects which MLP generates the map."""
        x_feat, _, edge_index, nt, et = make_graph()
        conv = make_conv(GeneralHetNLSDConv, ensemble=True, use_edge_types=False)
        # Warm-started generators are near-identical; perturb one so routing
        # through it is observable.
        with torch.no_grad():
            for param in conv.map_generators[0].parameters():
                param.add_(1.0)
            a, _ = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES, nt, et)
            b, _ = conv.get_restriction_maps(
                x_feat, edge_index, NUM_NODES, nt, torch.zeros_like(et)
            )
        assert not torch.allclose(a, b)

    def test_mlp_input_width_tracks_the_flags(self):
        conv = make_conv(DiagonalHetNLSDConv)
        expected = 2 * CONTEXT_DIM + 2 * NUM_NODE_TYPES + NUM_EDGE_TYPES
        assert conv._mlp_input_dim() == expected

    def test_ensemble_drops_edge_types_from_the_mlp_input(self):
        """In ensemble mode the edge type selects the MLP, it is not an input."""
        conv = make_conv(DiagonalHetNLSDConv, ensemble=True)
        assert conv._mlp_input_dim() == 2 * CONTEXT_DIM + 2 * NUM_NODE_TYPES


def diffusion_term(conv, x_feat, x_stalk, edge_index, nt, et):
    """Extracts the operator the layer applied, stripped of the skip path."""
    conv.eval()
    conv.sigma = nn.Identity()
    with torch.no_grad():
        return x_stalk - conv(x_feat, x_stalk, edge_index, nt, et)


class TestOperatorAssembly:
    """The het layer must still evaluate the node-wise Eq. 8 form."""

    def test_matches_the_node_wise_definition(self):
        x_feat, x_stalk, edge_index, nt, et = make_graph(seed=3, edge_prob=0.6)
        conv = make_conv(
            DiagonalHetNLSDConv,
            MLPNonlinearity(STALK_DIM, CHANNELS),
            add_eps=False,
            per_stalk_alpha=False,
        )
        diffusion = diffusion_term(conv, x_feat, x_stalk, edge_index, nt, et)

        with torch.no_grad():
            z = conv._apply_stalk_transform(x_stalk)
            src, dst = edge_index
            map_dst, map_src = conv.get_restriction_maps(
                x_feat, edge_index, NUM_NODES, nt, et
            )
            # sum_e F_v^T phi(F_v z_v - F_u z_u), one term per *directed* edge.
            local = map_dst[:, :, None] * z[dst] - map_src[:, :, None] * z[src]
            phi = conv.nonlinearity(local, x_feat[dst], x_feat[src])
            expected = torch.zeros_like(z)
            expected.index_add_(0, dst, map_dst[:, :, None] * phi)

        torch.testing.assert_close(diffusion, expected, atol=1e-5, rtol=1e-4)

    def test_add_self_loops_matches_passing_them_explicitly(self):
        """The add_self_loops path must agree with a pre-augmented edge_index.

        This pins the het-specific part of that path: ``edge_types`` has to be
        extended alongside ``edge_index``, otherwise the two disagree (or the
        one-hot lookup goes out of bounds).
        """
        x_feat, x_stalk, edge_index, nt, et = make_graph(seed=4)
        loops = torch.arange(NUM_NODES)
        with_loops = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
        et_with_loops = torch.cat([et, et.new_zeros(NUM_NODES)])

        conv = make_conv(DiagonalHetNLSDConv, IdentityNonlinearity())
        conv.eval()
        with torch.no_grad():
            explicit = conv(x_feat, x_stalk, with_loops, nt, et_with_loops)
            conv.add_self_loops = True
            automatic = conv(x_feat, x_stalk, edge_index, nt, et)
        torch.testing.assert_close(automatic, explicit, atol=1e-5, rtol=1e-4)


class TestHetNLSDConvValidation:
    def test_unknown_orth_strategy_raises(self):
        with pytest.raises(ValueError, match="orth_strategy must be one of"):
            make_conv(OrthogonalHetNLSDConv, orth_strategy="not_a_strategy")

    def test_non_positive_rank_raises(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            make_conv(LowRankHetNLSDConv, rank=0)

    def test_apply_norm_is_not_the_het_nlsd_path(self):
        """NLSD normalises endpoint maps, never assembled products."""
        conv = make_conv(DiagonalHetNLSDConv)
        with pytest.raises(NotImplementedError, match="get_restriction_maps"):
            conv._apply_norm(None, None, None, 0)

    def test_reset_parameters_restores_reference_init(self):
        conv = make_conv(
            DiagonalHetNLSDConv, ScalarThresholdNonlinearity(init_threshold=1.5)
        )
        with torch.no_grad():
            conv.eps.fill_(3.0)
            conv.alpha_gate.fill_(7.0)
            conv.nonlinearity.threshold.fill_(9.0)
        conv.reset_parameters()
        torch.testing.assert_close(conv.eps, torch.zeros(STALK_DIM))
        torch.testing.assert_close(conv.alpha_gate, -2.0 * torch.ones(STALK_DIM))
        torch.testing.assert_close(conv.nonlinearity.threshold, torch.tensor(1.5))
