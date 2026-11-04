# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Unit tests for the NLSD convolution layers."""

import random

import pytest
import torch
from torch import nn
from torch_geometric.utils import erdos_renyi_graph, to_undirected

from sheaf_mpnn.nlsd.nlsd_layers import (
    DiagonalNLSDConv,
    GeneralNLSDConv,
    LowRankNLSDConv,
    OrthogonalNLSDConv,
)
from sheaf_mpnn.nlsd.nonlinearity import (
    EdgeThresholdNonlinearity,
    IdentityNonlinearity,
    MLPNonlinearity,
    ScalarThresholdNonlinearity,
)
from sheaf_mpnn.nsd.nsd_layers import (
    DiagonalNSDConv,
    GeneralNSDConv,
    LowRankNSDConv,
    OrthogonalNSDConv,
)

NUM_NODES, STALK_DIM, CHANNELS, HIDDEN_DIM = 12, 3, 4, 8
CONTEXT_DIM = STALK_DIM * CHANNELS
RANK = 2


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_graph(seed=42, edge_prob=0.4):
    set_seed(seed)
    x_stalk = torch.randn(NUM_NODES, STALK_DIM, CHANNELS)
    x_feat = x_stalk.reshape(NUM_NODES, -1)
    # NLSD orients each undirected edge canonically, so the input must actually
    # be undirected for delta / delta^T to be adjoint.
    edge_index = to_undirected(erdos_renyi_graph(NUM_NODES, edge_prob=edge_prob))
    return x_feat, x_stalk, edge_index


def make_conv(conv_class, nonlinearity, **kwargs):
    return conv_class(
        STALK_DIM,
        CHANNELS,
        HIDDEN_DIM,
        nonlinearity=nonlinearity,
        context_dim=CONTEXT_DIM,
        **kwargs,
    )


@pytest.mark.parametrize(
    "conv_class,layer_kwargs",
    [
        (DiagonalNLSDConv, {}),
        (OrthogonalNLSDConv, {}),
        (OrthogonalNLSDConv, {"orth_strategy": "householder"}),
        (OrthogonalNLSDConv, {"use_edge_weights": True}),
        (GeneralNLSDConv, {}),
        (LowRankNLSDConv, {"rank": RANK}),
    ],
    ids=[
        "diagonal",
        "orthogonal",
        "orthogonal_householder",
        "orthogonal_edge_weights",
        "general",
        "low_rank",
    ],
)
@pytest.mark.parametrize(
    "make_phi",
    [
        lambda: IdentityNonlinearity(),
        lambda: ScalarThresholdNonlinearity(),
        lambda: ScalarThresholdNonlinearity(psi="cos"),
        lambda: MLPNonlinearity(STALK_DIM, CHANNELS),
    ],
    ids=["identity", "scalar_threshold", "scalar_threshold_cos", "mlp"],
)
class TestNLSDConvVariants:
    @pytest.fixture
    def setup(self, conv_class, layer_kwargs, make_phi):
        x_feat, x_stalk, edge_index = make_graph()
        conv = make_conv(conv_class, make_phi(), **layer_kwargs)
        return conv, x_feat, x_stalk, edge_index

    def test_output_shape(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        out = conv(x_feat, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        conv(x_feat, x_stalk, edge_index).pow(2).mean().backward()
        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0
        gen_weight = conv.map_generator[0].weight
        assert gen_weight.grad is not None and gen_weight.grad.abs().sum() > 0

    def test_empty_graph_is_a_pure_skip(self, setup):
        """No edges -> no diffusion term, so only the eps-gated skip survives."""
        conv, x_feat, x_stalk, _ = setup
        conv.eval()
        empty = torch.empty((2, 0), dtype=torch.long)
        out = conv(x_feat, x_stalk, empty)
        expected = (1 + torch.tanh(conv.eps)[None, :, None]) * x_stalk
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, expected)

    def test_determinism_in_eval(self, setup):
        conv, x_feat, x_stalk, edge_index = setup
        conv.eval()
        with torch.no_grad():
            first = conv(x_feat, x_stalk, edge_index)
            second = conv(x_feat, x_stalk, edge_index)
        torch.testing.assert_close(first, second)


def permute(conv, x_feat, x_stalk, edge_index, perm):
    """Runs ``conv`` on a relabelled copy of the graph, undoing the relabelling."""
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    conv.eval()
    with torch.no_grad():
        plain = conv(x_feat, x_stalk, edge_index)
        relabelled = conv(x_feat[perm], x_stalk[perm], inverse[edge_index])
    return plain[perm], relabelled


class TestPermutationEquivariance:
    """The layer must not depend on how the nodes happen to be numbered.

    Writing the operator as ``delta^T Phi delta`` forces an arbitrary
    orientation on each undirected edge, and for a ``Phi`` that is not odd the
    result depends on it.  The layer sidesteps that by evaluating the node-wise
    definition (Eq. 8) instead, so equivariance must hold for *every* ``Phi``,
    including the paper's learned MLP.
    """

    @pytest.mark.parametrize(
        "make_phi",
        [
            lambda: IdentityNonlinearity(),
            lambda: ScalarThresholdNonlinearity(),
            lambda: ScalarThresholdNonlinearity(psi="cos"),
            lambda: EdgeThresholdNonlinearity(CONTEXT_DIM),
            lambda: MLPNonlinearity(STALK_DIM, CHANNELS),
        ],
        ids=["identity", "scalar_threshold", "scalar_cos", "edge_threshold", "mlp"],
    )
    @pytest.mark.parametrize(
        "conv_class,layer_kwargs",
        [
            (DiagonalNLSDConv, {}),
            (OrthogonalNLSDConv, {}),
            (GeneralNLSDConv, {}),
            (LowRankNLSDConv, {"rank": RANK}),
        ],
        ids=["diag", "orth", "general", "low_rank"],
    )
    def test_relabelling_the_graph_does_not_change_the_output(
        self, conv_class, layer_kwargs, make_phi
    ):
        x_feat, x_stalk, edge_index = make_graph()
        conv = make_conv(conv_class, make_phi(), **layer_kwargs)
        set_seed(7)
        expected, actual = permute(
            conv, x_feat, x_stalk, edge_index, torch.randperm(NUM_NODES)
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "nlsd_class,nsd_class,layer_kwargs",
    [
        (DiagonalNLSDConv, DiagonalNSDConv, {}),
        (OrthogonalNLSDConv, OrthogonalNSDConv, {}),
        (OrthogonalNLSDConv, OrthogonalNSDConv, {"use_edge_weights": True}),
        (GeneralNLSDConv, GeneralNSDConv, {}),
        (LowRankNLSDConv, LowRankNSDConv, {"rank": RANK}),
    ],
    ids=["diagonal", "orthogonal", "orthogonal_edge_weights", "general", "low_rank"],
)
class TestIdentityPhiRecoversNSD:
    """Section 4.2: NSD is NLSD with Phi = identity.

    This pins down the whole construction at once -- the canonical orientation,
    the split coboundary normalisation, and the delta^T Phi delta assembly all
    have to be right for the two layers to agree numerically.

    For the general family the equality is modulo ``apply_general_norm``'s
    ``[-1, 1]`` clamp, which has no endpoint-map equivalent and so is absent
    from the coboundary path. The clamp is inert on symmetrically normalised
    products (see ``test_general_clamp_is_inert``), which is why the two still
    agree to float32 noise here.
    """

    def test_matches_linear_nsd(self, nlsd_class, nsd_class, layer_kwargs):
        x_feat, x_stalk, edge_index = make_graph()
        nlsd = make_conv(
            nlsd_class, IdentityNonlinearity(), per_stalk_alpha=False, **layer_kwargs
        )
        nsd = nsd_class(
            STALK_DIM,
            CHANNELS,
            HIDDEN_DIM,
            context_dim=CONTEXT_DIM,
            **layer_kwargs,
        )
        # Same weights on both sides: W1/W2, the map generator, eps and alpha.
        missing = nsd.load_state_dict(nlsd.state_dict(), strict=False)
        assert not missing.unexpected_keys
        nlsd.eval()
        nsd.eval()

        with torch.no_grad():
            torch.testing.assert_close(
                nlsd(x_feat, x_stalk, edge_index),
                nsd(x_feat, x_stalk, edge_index),
                atol=1e-5,
                rtol=1e-4,
            )


def diffusion_term(conv, x_feat, x_stalk, edge_index):
    """Extracts the operator the layer applied, stripped of the skip path.

    With ``add_eps=False``, ``per_stalk_alpha=False``, ``alpha=1`` and
    ``sigma=id`` the update is ``x - L^Phi x``, so the difference is exactly
    ``L^Phi x``.
    """
    conv.eval()
    conv.sigma = nn.Identity()
    with torch.no_grad():
        return x_stalk - conv(x_feat, x_stalk, edge_index)


class TestOperatorAssembly:
    """What operator the layer actually applies, spelled out from scratch."""

    def test_matches_the_node_wise_definition(self):
        """Eq. 8: every endpoint feeds Phi the disagreement as *it* sees it."""
        x_feat, x_stalk, edge_index = make_graph(seed=3, edge_prob=0.6)
        conv = make_conv(
            DiagonalNLSDConv,
            MLPNonlinearity(STALK_DIM, CHANNELS),
            add_eps=False,
            per_stalk_alpha=False,
        )
        diffusion = diffusion_term(conv, x_feat, x_stalk, edge_index)

        with torch.no_grad():
            z = conv._apply_stalk_transform(x_stalk)
            src, dst = edge_index
            map_dst, map_src = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES)

            # sum_e F_v^T phi(F_v z_v - F_u z_u), one term per *directed* edge.
            local = map_dst[:, :, None] * z[dst] - map_src[:, :, None] * z[src]
            phi = conv.nonlinearity(local, x_feat[dst], x_feat[src])
            expected = torch.zeros_like(z)
            expected.index_add_(0, dst, map_dst[:, :, None] * phi)

        torch.testing.assert_close(diffusion, expected, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize(
        "make_phi",
        [
            lambda: IdentityNonlinearity(),
            lambda: ScalarThresholdNonlinearity(),
            lambda: ScalarThresholdNonlinearity(psi="cos"),
        ],
        ids=["identity", "scalar_threshold", "scalar_cos"],
    )
    def test_odd_phi_also_matches_delta_t_phi_delta(self, make_phi):
        """For odd Phi the node-wise form and the operator form coincide.

        This is the condition under which Eq. 8 and Eq. 11-12 of the paper are
        the same object -- and hence where this layer agrees with the reference
        implementation, which fixes one orientation per edge.
        """
        x_feat, x_stalk, edge_index = make_graph(seed=3, edge_prob=0.6)
        conv = make_conv(
            DiagonalNLSDConv, make_phi(), add_eps=False, per_stalk_alpha=False
        )
        diffusion = diffusion_term(conv, x_feat, x_stalk, edge_index)

        with torch.no_grad():
            z = conv._apply_stalk_transform(x_stalk)
            src, dst = edge_index
            map_dst, map_src = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES)

            # One row-block per undirected edge, oriented low -> high index.
            keep = src < dst
            low, high = src[keep], dst[keep]
            map_low, map_high = map_src[keep], map_dst[keep]

            delta = map_low[:, :, None] * z[low] - map_high[:, :, None] * z[high]
            phi = conv.nonlinearity(delta, x_feat[low], x_feat[high])
            expected = torch.zeros_like(z)
            expected.index_add_(0, low, map_low[:, :, None] * phi)
            expected.index_add_(0, high, -(map_high[:, :, None] * phi))

        torch.testing.assert_close(diffusion, expected, atol=1e-5, rtol=1e-4)


class TestMatrixMapVariants:
    """The two full-matrix families: general and low-rank."""

    @pytest.mark.parametrize(
        "conv_class,layer_kwargs",
        [(GeneralNLSDConv, {}), (LowRankNLSDConv, {"rank": RANK})],
        ids=["general", "low_rank"],
    )
    def test_matches_the_node_wise_definition(self, conv_class, layer_kwargs):
        """Eq. 8 spelled out by hand, with matrix maps instead of diagonal ones."""
        x_feat, x_stalk, edge_index = make_graph(seed=3, edge_prob=0.6)
        conv = make_conv(
            conv_class,
            MLPNonlinearity(STALK_DIM, CHANNELS),
            add_eps=False,
            per_stalk_alpha=False,
            **layer_kwargs,
        )
        diffusion = diffusion_term(conv, x_feat, x_stalk, edge_index)

        with torch.no_grad():
            z = conv._apply_stalk_transform(x_stalk)
            src, dst = edge_index
            map_dst, map_src = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES)

            # sum_e F_v^T phi(F_v z_v - F_u z_u), one term per *directed* edge.
            local = map_dst @ z[dst] - map_src @ z[src]
            phi = conv.nonlinearity(local, x_feat[dst], x_feat[src])
            expected = torch.zeros_like(z)
            expected.index_add_(0, dst, map_dst.transpose(-2, -1) @ phi)

        torch.testing.assert_close(diffusion, expected, atol=1e-5, rtol=1e-4)

    def test_low_rank_maps_respect_the_rank_bound(self):
        """The whole point of the family: F = A B^T has rank at most ``rank``."""
        x_feat, _, edge_index = make_graph()
        conv = make_conv(LowRankNLSDConv, IdentityNonlinearity(), rank=1)
        conv.eval()
        with torch.no_grad():
            map_dst, map_src = conv.get_restriction_maps(x_feat, edge_index, NUM_NODES)
        assert map_dst.shape == (edge_index.size(1), STALK_DIM, STALK_DIM)
        # Normalisation multiplies on the right, which cannot raise the rank.
        # rtol keeps float32 round-off out of the singular-value count.
        for maps in (map_dst, map_src):
            assert int(torch.linalg.matrix_rank(maps, rtol=1e-4).max()) <= 1

    def test_rank_deficient_degree_stays_finite(self):
        """A rank bound with ``rank * deg(v) < d`` makes the sheaf degree singular.

        Only the augmented ``(D+I)^{-1/2}`` keeps the normalisation defined
        there -- a path graph with rank 1 and d = 3 is the smallest case.
        """
        set_seed(11)
        x_stalk = torch.randn(4, STALK_DIM, CHANNELS)
        x_feat = x_stalk.reshape(4, -1)
        path = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        conv = make_conv(LowRankNLSDConv, MLPNonlinearity(STALK_DIM, CHANNELS), rank=1)
        out = conv(x_feat, x_stalk, path)
        assert torch.isfinite(out).all()
        out.pow(2).mean().backward()
        assert torch.isfinite(conv.map_generator[0].weight.grad).all()

    def test_general_clamp_is_inert(self):
        """Why dropping ``apply_general_norm``'s clamp changes nothing.

        Symmetric ``(D+I)^{-1/2}`` normalisation already leaves every entry of
        the assembled products inside ``[-1, 1]``, so the clamp the linear path
        applies -- and which has no endpoint-map equivalent -- never fires.
        """
        for seed, edge_prob in ((0, 0.2), (1, 0.5), (2, 0.9)):
            x_feat, _, edge_index = make_graph(seed=seed, edge_prob=edge_prob)
            conv = make_conv(GeneralNLSDConv, IdentityNonlinearity())
            conv.eval()
            with torch.no_grad():
                map_dst, map_src = conv.get_restriction_maps(
                    x_feat, edge_index, NUM_NODES
                )
                self_map = map_dst.transpose(-2, -1) @ map_dst
                cross_map = map_dst.transpose(-2, -1) @ map_src
            assert self_map.abs().max() <= 1.0
            assert cross_map.abs().max() <= 1.0

    def test_non_positive_rank_raises(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            make_conv(LowRankNLSDConv, IdentityNonlinearity(), rank=0)


class TestPerStalkAlphaGate:
    def test_gate_scales_the_diffusion_term(self):
        x_feat, x_stalk, edge_index = make_graph()
        conv = make_conv(
            DiagonalNLSDConv, ScalarThresholdNonlinearity(), per_stalk_alpha=True
        )
        conv.eval()
        with torch.no_grad():
            gated = conv(x_feat, x_stalk, edge_index)
            # Push the gate to zero: the update collapses to the pure skip.
            conv.alpha_gate.fill_(-50.0)
            skip_only = conv(x_feat, x_stalk, edge_index)
        expected = (1 + torch.tanh(conv.eps)[None, :, None]) * x_stalk
        torch.testing.assert_close(skip_only, expected)
        assert not torch.allclose(gated, skip_only)

    def test_absent_when_disabled(self):
        conv = make_conv(
            DiagonalNLSDConv, ScalarThresholdNonlinearity(), per_stalk_alpha=False
        )
        assert not hasattr(conv, "alpha_gate")

    def test_reset_parameters_restores_reference_init(self):
        conv = make_conv(DiagonalNLSDConv, ScalarThresholdNonlinearity())
        with torch.no_grad():
            conv.alpha_gate.fill_(3.0)
            conv.eps.fill_(3.0)
        conv.reset_parameters()
        torch.testing.assert_close(conv.alpha_gate, torch.full((STALK_DIM,), -2.0))
        torch.testing.assert_close(conv.eps, torch.zeros(STALK_DIM))


class TestNLSDConvValidation:
    def test_unknown_orth_strategy_raises(self):
        with pytest.raises(ValueError, match="orth_strategy must be one of"):
            make_conv(
                OrthogonalNLSDConv, IdentityNonlinearity(), orth_strategy="nonsense"
            )

    def test_apply_norm_is_not_the_nlsd_path(self):
        """The product-based normalisation of the linear families is unusable."""
        conv = make_conv(DiagonalNLSDConv, IdentityNonlinearity())
        with pytest.raises(NotImplementedError, match="get_restriction_maps"):
            conv._apply_norm(None, None, None, None)

    @pytest.mark.parametrize(
        "make_phi",
        [
            lambda: IdentityNonlinearity(),
            lambda: ScalarThresholdNonlinearity(),
            lambda: MLPNonlinearity(STALK_DIM, CHANNELS),
        ],
        ids=["identity", "scalar_threshold", "mlp"],
    )
    def test_self_loops_carry_no_diffusion(self, make_phi):
        """A self-loop's coboundary is F x - F x = 0, and every Phi maps 0 to 0.

        A graph of nothing but self-loops therefore diffuses nothing, and the
        update collapses to the eps-gated skip term.
        """
        x_feat, x_stalk, _ = make_graph()
        conv = make_conv(DiagonalNLSDConv, make_phi())
        conv.eval()
        loops = torch.arange(NUM_NODES).repeat(2, 1)
        with torch.no_grad():
            out = conv(x_feat, x_stalk, loops)
        expected = (1 + torch.tanh(conv.eps)[None, :, None]) * x_stalk
        torch.testing.assert_close(out, expected)
