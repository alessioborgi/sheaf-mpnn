# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""The BuNN layer: heat-kernel solvers, and the flat-bundle factorisation.

The load-bearing claim is Lemma 3.1 -- that on a *flat* bundle the bundle heat
kernel factors as :math:`\\mathcal{H}_\\mathcal{B}(t) = O^\\top \\mathcal{H}(t)
O`, which is what lets the layer diffuse cheap scalar channels instead of
exponentiating an :math:`nd \\times nd` operator. ``test_matches_the_explicit_
bundle_heat_kernel`` checks it against a brute-force ``matrix_exp``.
"""

import pytest
import torch

from sheaf_mpnn.bunn import BuNNConv, laplacian_eigendecomposition, sync
from sheaf_mpnn.bunn.bunn_layers import TAYLOR_MAX_TAU, stable_taylor_degree
from tests.bunn.test_bunn_bundle import ring_edge_index

NUM_NODES = 8
NUM_BUNDLES = 3
BUNDLE_DIM = 2
CHANNELS = 4
WIDTH = NUM_BUNDLES * BUNDLE_DIM * CHANNELS


@pytest.fixture
def edge_index() -> torch.Tensor:
    return ring_edge_index(NUM_NODES)


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(NUM_NODES, WIDTH)


def make_conv(**kwargs) -> BuNNConv:
    defaults = {
        "num_bundles": NUM_BUNDLES,
        "bundle_dim": BUNDLE_DIM,
        "in_channels": CHANNELS,
    }
    defaults.update(kwargs)
    return BuNNConv(**defaults)  # ty: ignore[invalid-argument-type]


class TestForward:
    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    def test_output_shape_and_finiteness(self, solver, x, edge_index):
        out = make_conv(solver=solver)(x, x, edge_index)
        assert out.shape == (NUM_NODES, WIDTH)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    def test_gradients_reach_every_parameter(self, solver, x, edge_index):
        conv = make_conv(solver=solver, learn_tau=True)
        conv(x, x, edge_index).sum().backward()
        assert all(p.grad is not None for p in conv.parameters())

    def test_zero_diffusion_time_is_the_encoder_alone(self, x, edge_index):
        r"""At t = 0 the heat kernel is the identity, so the layer reduces to
        Algorithm 1's encoder :math:`O_v^\top (W O_v x_v + b)`.

        Written out per node with plain tensor ops rather than the layer's own
        ``sync``/``desync`` helpers -- a check built from those would restate
        the implementation instead of testing it, and in particular could not
        notice the bias landing on the wrong side of a frame change.
        """
        conv = make_conv(tau=0.0)
        assert conv.map_learner is not None
        maps = conv.map_learner(x, edge_index)
        W, b = conv.linear.weight, conv.linear.bias

        got = conv(x, x, edge_index)

        for v in range(NUM_NODES):
            # Build node v's full block-diagonal frame O_v over all bundles.
            frame = torch.block_diag(
                *[
                    torch.kron(maps[v, k], torch.eye(CHANNELS))
                    for k in range(NUM_BUNDLES)
                ]
            )
            expected = frame.T @ (W @ (frame @ x[v]) + b)
            assert torch.allclose(got[v], expected, atol=1e-5)

    def test_encoder_bias_is_rotated_not_added_in_the_local_frame(self, x, edge_index):
        r"""Pins the one place this deviates from Eq. 2 as written.

        Algorithm 1 applies ``b`` in the synchronised frame, so it reaches the
        output as :math:`O_v^\top b`, not the :math:`+\,b` of Eq. 2. The two
        coincide only for a trivial bundle. This test exists so the choice
        cannot be changed silently in either direction.
        """
        conv = make_conv(tau=0.0)
        assert conv.map_learner is not None
        maps = conv.map_learner(x, edge_index)
        W, b = conv.linear.weight, conv.linear.bias
        got = conv(x, x, edge_index)

        frames = [
            torch.block_diag(
                *[
                    torch.kron(maps[v, k], torch.eye(CHANNELS))
                    for k in range(NUM_BUNDLES)
                ]
            )
            for v in range(NUM_NODES)
        ]
        rotated = torch.stack(
            [f.T @ (W @ (f @ x[v]) + b) for v, f in enumerate(frames)]
        )
        eq2_literal = torch.stack(
            [f.T @ W @ (f @ x[v]) + b for v, f in enumerate(frames)]
        )

        assert torch.allclose(got, rotated, atol=1e-5)
        assert not torch.allclose(got, eq2_literal, atol=1e-3)

    def test_permutation_equivariance(self, x, edge_index):
        conv = make_conv(tau=0.5)
        perm = torch.randperm(NUM_NODES)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(NUM_NODES)

        out = conv(x, x, edge_index)
        permuted = conv(x[perm], x[perm], inverse[edge_index])
        assert torch.allclose(permuted, out[perm], atol=1e-5)

    def test_external_maps_require_node_rep(self, x, edge_index):
        conv = make_conv(external_maps=True)
        assert conv.map_learner is None
        with pytest.raises(ValueError, match="external_maps"):
            conv(x, x, edge_index)

    def test_external_maps_are_used_when_supplied(self, x, edge_index):
        conv = make_conv(external_maps=True)
        maps = torch.eye(BUNDLE_DIM).expand(
            NUM_NODES, NUM_BUNDLES, BUNDLE_DIM, BUNDLE_DIM
        )
        out = conv(x, x, edge_index, node_rep=maps)
        assert out.shape == (NUM_NODES, WIDTH)

    def test_wrong_width_is_rejected(self, edge_index):
        conv = make_conv()
        with pytest.raises(ValueError, match="num_bundles \\* bundle_dim"):
            conv(
                torch.randn(NUM_NODES, WIDTH + 1),
                torch.randn(NUM_NODES, WIDTH + 1),
                edge_index,
            )


class TestHeatKernel:
    def test_taylor_and_spectral_agree_for_small_time(self, x, edge_index):
        """The paper's two algorithms compute the same operator; at small t a
        high-degree truncation should reproduce the exact spectral answer.
        """
        taylor = make_conv(tau=0.3, solver="taylor", max_degree=20)
        spectral = make_conv(tau=0.3, solver="spectral")
        spectral.load_state_dict(taylor.state_dict())

        assert torch.allclose(
            taylor(x, x, edge_index), spectral(x, x, edge_index), atol=1e-5
        )

    def test_the_papers_time_grid_is_either_accurate_or_refused(self, x, edge_index):
        """Every entry of the paper's 0.1/1/10/100 grid must be safe to run.

        Where Taylor is admissible it has to match the exact spectral answer;
        where it is not, construction has to fail. Before the guard the t >= 10
        entries silently returned values ~1e5 and ~1e13 times too large, which
        reads as diverged training rather than a misused solver.
        """
        for tau in (0.1, 1.0, 10.0, 100.0):
            spectral = make_conv(tau=tau, solver="spectral")
            assert torch.isfinite(spectral(x, x, edge_index)).all()

            if tau > TAYLOR_MAX_TAU:
                with pytest.raises(ValueError, match="Taylor solver can evaluate"):
                    make_conv(tau=tau, solver="taylor", max_degree=1000)
                continue

            taylor = make_conv(
                tau=tau, solver="taylor", max_degree=stable_taylor_degree(tau)
            )
            spectral.load_state_dict(taylor.state_dict())
            got, want = taylor(x, x, edge_index), spectral(x, x, edge_index)
            assert torch.isfinite(got).all()
            assert torch.allclose(got, want, rtol=0, atol=1e-2), (
                f"tau={tau}: max diff {(got - want).abs().max().item():g}"
            )

    def test_matches_the_explicit_bundle_heat_kernel(self, x, edge_index):
        r"""Lemma 3.1, checked against a brute-force ``exp(-t L_B)``.

        Builds the full :math:`nd \times nd` bundle Laplacian
        :math:`L_\mathcal{B} = O^\top (\mathcal{L} \otimes I_d) O` for one
        bundle and exponentiates it directly.
        """
        tau = 0.4
        conv = make_conv(tau=tau, solver="spectral")
        assert conv.map_learner is not None
        maps = conv.map_learner(x, edge_index)

        h = conv.linear(sync(conv._to_bundle(x), maps).reshape(NUM_NODES, WIDTH))
        got = conv(x, x, edge_index)

        eig_vals, eig_vecs = laplacian_eigendecomposition(edge_index, NUM_NODES)
        lap = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T

        for bundle in range(NUM_BUNDLES):
            block = torch.block_diag(*[maps[i, bundle] for i in range(NUM_NODES)])
            lap_bundle = block.T @ torch.kron(lap, torch.eye(BUNDLE_DIM)) @ block
            kernel = torch.matrix_exp(-tau * lap_bundle)

            for channel in range(CHANNELS):
                # The layer synchronises, diffuses, desynchronises; equivalently
                # the bundle kernel acts on the desynchronised encoded signal.
                encoded = h.view(NUM_NODES, NUM_BUNDLES, BUNDLE_DIM, CHANNELS)[
                    :, bundle, :, channel
                ].reshape(-1)
                expected = kernel @ (block.T @ encoded)
                actual = got.view(NUM_NODES, NUM_BUNDLES, BUNDLE_DIM, CHANNELS)[
                    :, bundle, :, channel
                ].reshape(-1)
                assert torch.allclose(actual, expected, atol=1e-5)

    def test_trivial_bundle_diffusion_smooths_towards_the_degree_profile(self):
        """With identity maps the bundle is trivial and BuNN degenerates to
        scalar heat diffusion -- the over-smoothing regime the bundle exists
        to escape (Prop. 4.1). Long diffusion must flatten the signal.
        """
        edge_index = ring_edge_index(NUM_NODES)
        conv = BuNNConv(1, 2, 1, tau=50.0, solver="spectral", external_maps=True)
        with torch.no_grad():
            conv.linear.weight.copy_(torch.eye(2))
            conv.linear.bias.zero_()

        maps = torch.eye(2).expand(NUM_NODES, 1, 2, 2)
        x = torch.randn(NUM_NODES, 2)
        out = conv(x, x, edge_index, node_rep=maps)

        spread = out.std(dim=0).max()
        assert spread < 1e-3

    def test_every_node_pair_communicates_in_one_layer(self):
        """The paper's key structural claim (Lemma 4.3): H(t, u, v) > 0 for all
        pairs, so a single layer has no under-reaching -- unlike message
        passing, where a node 3 hops away cannot influence the output at all.
        """
        num_nodes = 6
        # Path graph: nodes 0 and 5 are five hops apart.
        src = torch.arange(num_nodes - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

        conv = BuNNConv(1, 2, 1, tau=1.0, solver="spectral", external_maps=True)
        maps = torch.eye(2).expand(num_nodes, 1, 2, 2)
        x = torch.zeros(num_nodes, 2, requires_grad=True)

        out = conv(x, x, edge_index, node_rep=maps)
        out[0].sum().backward()

        assert x.grad is not None
        assert x.grad[num_nodes - 1].abs().sum() > 0


class TestNormalization:
    @pytest.mark.parametrize("normalize", ["sym", "rw"])
    def test_taylor_supports_both_normalizations(self, normalize, x, edge_index):
        out = make_conv(normalize=normalize)(x, x, edge_index)
        assert torch.isfinite(out).all()

    def test_random_walk_diffusion_preserves_constants(self, edge_index):
        """The rw Laplacian kills constant signals, so heat flow fixes them."""
        conv = BuNNConv(
            1, 2, 1, tau=4.0, normalize="rw", external_maps=True, max_degree=32
        )
        with torch.no_grad():
            conv.linear.weight.copy_(torch.eye(2))
            conv.linear.bias.zero_()

        maps = torch.eye(2).expand(NUM_NODES, 1, 2, 2)
        x = torch.ones(NUM_NODES, 2)
        out = conv(x, x, edge_index, node_rep=maps)
        assert torch.allclose(out, x, atol=1e-5)

    def test_isolated_nodes_do_not_produce_nans(self):
        """Degree-0 nodes would divide by zero in either normalisation."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        for normalize in ("sym", "rw"):
            conv = BuNNConv(2, 2, 2, normalize=normalize)
            x = torch.randn(4, 8)  # node 2 and 3 are isolated
            assert torch.isfinite(conv(x, x, edge_index)).all()


class TestConstruction:
    def test_learned_tau_is_per_bundle_and_non_negative(self, x, edge_index):
        conv = make_conv(learn_tau=True, tau=2.0, max_degree=16)
        assert isinstance(conv.tau, torch.nn.Parameter)
        assert conv.tau.shape == (NUM_BUNDLES,)

        # A negative learned t is clamped by the ReLU, not propagated.
        with torch.no_grad():
            conv.tau.fill_(-5.0)
        out = conv(x, x, edge_index)
        assert torch.isfinite(out).all()

    def test_fixed_tau_is_a_buffer_not_a_parameter(self):
        conv = make_conv(learn_tau=False, tau=1.5, max_degree=12)
        assert "tau" in dict(conv.named_buffers())
        assert "tau" not in dict(conv.named_parameters())

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"num_bundles": 0}, "num_bundles"),
            ({"bundle_dim": 0}, "bundle_dim"),
            ({"in_channels": 0}, "in_channels"),
            ({"tau": -1.0}, "tau"),
            ({"max_degree": 0}, "max_degree"),
            ({"solver": "euler"}, "solver"),
            ({"normalize": "none"}, "normalize"),
            ({"solver": "spectral", "normalize": "rw"}, "spectral solver requires"),
            ({"tau": 3.0, "max_degree": 8}, "cannot represent"),
            ({"tau": 10.0}, "Taylor solver can evaluate"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_conv(**kwargs)

    def test_reset_parameters_reinitialises(self, x, edge_index):
        conv = make_conv(learn_tau=True)
        before = conv.linear.weight.clone()
        conv.reset_parameters()
        assert not torch.allclose(before, conv.linear.weight)


class TestTaylorStabilityGuard:
    """A too-short truncation of ``exp(-t L)`` diverges; it must not run.

    The terms ``(t*lambda)^k / k!`` peak at ``k ~ t*lambda``, so truncating
    below the peak returns a large positive number where a small decaying one
    belongs. That looks like diverged training, not a solver misuse, so the
    layer refuses to be built instead of computing it.
    """

    def test_the_default_pairing_is_admitted(self):
        """tau=1 with degree 8 is the paper's own default and must stay legal."""
        assert stable_taylor_degree(1.0) <= 8
        make_conv(tau=1.0, solver="taylor", max_degree=8)

    @pytest.mark.parametrize("tau", [10.0, 100.0])
    def test_times_beyond_floating_point_reach_are_refused_at_any_degree(self, tau):
        """No truncation rescues these: the partial sums reach e^{t*lambda}
        while the answer is e^{-t*lambda}, so it is all cancellation.
        """
        for degree in (8, 1000):
            with pytest.raises(ValueError, match="Taylor solver can evaluate"):
                make_conv(tau=tau, solver="taylor", max_degree=degree)

    def test_too_small_a_degree_is_refused_within_the_usable_range(self):
        with pytest.raises(ValueError, match="cannot represent"):
            make_conv(tau=3.0, solver="taylor", max_degree=8)

    @pytest.mark.parametrize("tau", [0.1, 0.5, 1.0, 2.0, TAYLOR_MAX_TAU])
    def test_the_recommended_degree_is_actually_accurate(self, tau, x, edge_index):
        """The bound must be honest: at the degree it prescribes, the series
        has to match the exact kernel, not merely avoid overflow.
        """
        conv = make_conv(tau=tau, solver="taylor", max_degree=stable_taylor_degree(tau))
        exact = make_conv(tau=tau, solver="spectral")
        exact.load_state_dict(conv.state_dict())
        assert torch.allclose(
            conv(x, x, edge_index), exact(x, x, edge_index), rtol=0, atol=1e-2
        )

    def test_spectral_is_never_restricted_by_the_bound(self, x, edge_index):
        conv = make_conv(tau=100.0, solver="spectral")
        assert torch.isfinite(conv(x, x, edge_index)).all()

    def test_learned_time_is_capped_at_what_the_degree_can_represent(
        self, x, edge_index
    ):
        """Training could otherwise walk tau into the divergent region after
        construction, where the guard can no longer see it.
        """
        conv = make_conv(tau=1.0, solver="taylor", max_degree=8, learn_tau=True)
        with torch.no_grad():
            conv.tau.fill_(500.0)

        out = conv(x, x, edge_index)
        assert torch.isfinite(out).all()
        assert out.abs().max() < 1e3
        assert conv.tau_cap == pytest.approx(1.0)

    def test_the_cap_never_exceeds_the_floating_point_limit(self):
        conv = make_conv(tau=1.0, solver="taylor", max_degree=1000, learn_tau=True)
        assert conv.tau_cap == pytest.approx(TAYLOR_MAX_TAU)


class TestLaplacianEigendecomposition:
    def test_reconstructs_the_normalised_laplacian(self, edge_index):
        eig_vals, eig_vecs = laplacian_eigendecomposition(edge_index, NUM_NODES)
        lap = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T

        # A ring is 2-regular, so L = I - A/2.
        adj = torch.zeros(NUM_NODES, NUM_NODES)
        adj[edge_index[1], edge_index[0]] = 1.0
        assert torch.allclose(lap, torch.eye(NUM_NODES) - adj / 2, atol=1e-5)

    def test_directed_input_is_rejected_rather_than_silently_symmetrised(self):
        """``eigh`` reads one triangle and assumes the other mirrors it.

        On a directed edge_index that assumption is false, so it would return
        the spectrum of a graph the caller never passed -- with no error. The
        repo exposes ``--directed``, so this is reachable from the CLI.
        """
        directed = torch.tensor([[0, 1], [1, 2]])  # 0 -> 1 -> 2, one direction
        with pytest.raises(ValueError, match="undirected"):
            laplacian_eigendecomposition(directed, 3)

    def test_undirected_input_is_accepted(self):
        undirected = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        eig_vals, _ = laplacian_eigendecomposition(undirected, 3)
        assert eig_vals.shape == (3,)

    def test_oversized_graphs_are_refused_before_allocating(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="max_dense_nodes"):
            laplacian_eigendecomposition(edge_index, 50_000, max_dense_nodes=1_000)

    def test_parallel_edges_accumulate(self):
        """Duplicate entries must sum into the adjacency, not overwrite it."""
        single = torch.tensor([[0, 1], [1, 0]])
        doubled = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        # Doubling every edge doubles the degree too, so the normalised
        # Laplacian is unchanged -- which only holds if duplicates accumulate.
        a, _ = laplacian_eigendecomposition(single, 2)
        b, _ = laplacian_eigendecomposition(doubled, 2)
        assert torch.allclose(a, b, atol=1e-6)

    @pytest.mark.parametrize("num_eigs", [1, 3, 8])
    def test_truncation_returns_the_low_frequency_end(self, num_eigs, edge_index):
        full_vals, _ = laplacian_eigendecomposition(edge_index, NUM_NODES)
        vals, vecs = laplacian_eigendecomposition(
            edge_index, NUM_NODES, num_eigs=num_eigs
        )
        assert vals.shape == (num_eigs,)
        assert vecs.shape == (NUM_NODES, num_eigs)
        assert torch.allclose(vals, full_vals[:num_eigs], atol=1e-6)

    def test_truncation_range_is_validated(self, edge_index):
        for bad in (0, NUM_NODES + 1):
            with pytest.raises(ValueError, match="num_eigs"):
                laplacian_eigendecomposition(edge_index, NUM_NODES, num_eigs=bad)

    def test_full_truncation_is_the_exact_solver(self, x, edge_index):
        truncated = make_conv(tau=1.0, solver="spectral", num_eigs=NUM_NODES)
        exact = make_conv(tau=1.0, solver="spectral")
        truncated.load_state_dict(exact.state_dict())
        assert torch.allclose(
            truncated(x, x, edge_index), exact(x, x, edge_index), atol=1e-5
        )

    def test_truncation_approximates_long_diffusions_well(self, x, edge_index):
        """exp(-t*lambda) suppresses the high end, so dropping it costs little
        at large t -- the reason the paper truncates at all.
        """
        exact = make_conv(tau=20.0, solver="spectral")
        truncated = make_conv(tau=20.0, solver="spectral", num_eigs=2)
        truncated.load_state_dict(exact.state_dict())
        # Keeping 2 of 8 eigenpairs still tracks the exact kernel to ~1e-3,
        # because exp(-20*lambda) has already annihilated the rest.
        assert torch.allclose(
            truncated(x, x, edge_index), exact(x, x, edge_index), rtol=0, atol=5e-3
        )

    def test_spectrum_lies_in_the_expected_range(self, edge_index):
        eig_vals, _ = laplacian_eigendecomposition(edge_index, NUM_NODES)
        assert eig_vals.min() > -1e-5
        assert eig_vals.max() < 2 + 1e-5

    def test_full_spectrum_above_the_cap_is_still_refused(self):
        """Only a truncated request escapes the dense cap; the full spectrum
        has no sparse shortcut, so a large graph without num_eigs still errors.
        """
        edges = ring_edge_index(40)
        with pytest.raises(ValueError, match="max_dense_nodes"):
            laplacian_eigendecomposition(edges, 40, max_dense_nodes=10)

    def test_truncated_spectrum_above_the_cap_matches_the_dense_solver(self):
        """A truncated request past the dense cap is served sparsely -- and the
        sparse low-frequency eigenpairs must agree with the dense solver's.

        ``k = 5`` lands on a ring's eigenspace boundary (the 0 eigenvalue plus
        two degenerate pairs), so the two 5-dim subspaces coincide and the
        truncated heat operator is basis-independent -- comparable despite the
        degeneracy that makes raw eigenvectors ambiguous.
        """
        n, k = 40, 5
        edges = ring_edge_index(n)
        dense_vals, dense_vecs = laplacian_eigendecomposition(edges, n, num_eigs=k)
        # max_dense_nodes below n forces the sparse path on the same small graph.
        sparse_vals, sparse_vecs = laplacian_eigendecomposition(
            edges, n, num_eigs=k, max_dense_nodes=10
        )

        assert sparse_vals.shape == (k,)
        assert sparse_vecs.shape == (n, k)
        # Ascending eigenvalues match the dense low-frequency end.
        assert torch.allclose(sparse_vals, dense_vals, atol=1e-4)
        # Columns are orthonormal.
        assert torch.allclose(sparse_vecs.T @ sparse_vecs, torch.eye(k), atol=1e-4)
        # Rotation-invariant check: the rank-k heat operator V exp(-t L) V^T
        # agrees, so the layer diffuses identically under either solver.
        t = 1.5
        op_dense = dense_vecs @ torch.diag(torch.exp(-t * dense_vals)) @ dense_vecs.T
        op_sparse = (
            sparse_vecs @ torch.diag(torch.exp(-t * sparse_vals)) @ sparse_vecs.T
        )
        assert torch.allclose(op_dense, op_sparse, atol=1e-4)
