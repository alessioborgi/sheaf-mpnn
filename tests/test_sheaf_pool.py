# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for SheafPool (SHEAFPOOL) and its utility functions.

Covers:
  - Output shape and numerical sanity
  - Permutation invariance (within each graph)
  - Exact O(d) gauge invariance
  - Batch handling (single graph, multi-graph, no-batch alias)
  - Both input formats: [N, d, C] and [N, d*C]
  - Gradient flow through the full pipeline
  - Parametrised over (stalk_dim, channel_dim) combinations
  - Utility: batched_sym_matrix_pow (p=-0.5), batched_procrustes_align
"""

import pytest
import torch

from sheaf_mpnn.sheaf_pool import SheafPool
from sheaf_mpnn.utils import batched_procrustes_align, batched_sym_matrix_pow

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_orth(n: int, d: int) -> torch.Tensor:
    """Return n random orthogonal d x d matrices (det may be +/-1)."""
    A = torch.randn(n, d, d)
    Q, _ = torch.linalg.qr(A)
    return Q  # [n, d, d], each Q[i] in O(d)


def _random_so(n: int, d: int) -> torch.Tensor:
    """Return n random *proper* rotation d x d matrices (det = +1)."""
    Q = _random_orth(n, d)
    # Flip last column of any Q with det = -1 to force det = +1
    det = torch.linalg.det(Q)  # [n]
    flip = torch.ones(n, d, device=Q.device, dtype=Q.dtype)
    flip[:, -1] = det.sign()
    Q = Q * flip[:, None, :]
    return Q


def _make_pool(d: int, C: int, **kw) -> SheafPool:
    pool = SheafPool(stalk_dim=d, channel_dim=C, **kw)
    pool.eval()
    return pool


def _make_embeddings(N: int, d: int, C: int, seed: int = 42) -> torch.Tensor:
    """Random [N, d, C] stalk embeddings."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(N, d, C, generator=g)


# ---------------------------------------------------------------------------
# Utility: batched_sym_matrix_pow (p=-0.5, replaces batched_inv_sqrt_spd)
# ---------------------------------------------------------------------------


class TestBatchedSymMatrixPowInvSqrt:
    """Tests for batched_sym_matrix_pow with p=-0.5 (replaces batched_inv_sqrt_spd)."""

    def test_identity_matrix(self):
        B, d = 4, 5
        eye = torch.eye(d).unsqueeze(0).expand(B, -1, -1)
        out = batched_sym_matrix_pow(eye, p=-0.5)
        torch.testing.assert_close(out, eye, atol=1e-5, rtol=1e-5)

    def test_inverse_sqrt_property(self):
        """(S^{-1/2})^2 = S^{-1} up to tolerance."""
        B, d = 6, 4
        A = torch.randn(B, d, d)
        S = A @ A.mT + torch.eye(d).unsqueeze(0) * 0.5  # SPD
        S_inv_sqrt = batched_sym_matrix_pow(S, p=-0.5, eps=1e-6)
        S_inv_approx = S_inv_sqrt @ S_inv_sqrt
        S_inv_true = torch.linalg.inv(S)
        torch.testing.assert_close(S_inv_approx, S_inv_true, atol=1e-4, rtol=1e-4)

    def test_output_is_symmetric(self):
        B, d = 5, 3
        A = torch.randn(B, d, d)
        S = A @ A.mT + torch.eye(d)
        out = batched_sym_matrix_pow(S, p=-0.5)
        torch.testing.assert_close(out, out.mT, atol=1e-6, rtol=1e-6)

    def test_output_shape(self):
        B, d = 7, 6
        S = torch.eye(d).unsqueeze(0).expand(B, -1, -1).clone()
        assert batched_sym_matrix_pow(S, p=-0.5).shape == (B, d, d)

    def test_no_nan(self):
        B, d = 4, 4
        A = torch.randn(B, d, d)
        S = A @ A.mT + 0.1 * torch.eye(d)
        out = batched_sym_matrix_pow(S, p=-0.5, eps=1e-6)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Utility: batched_procrustes_align
# ---------------------------------------------------------------------------


class TestBatchedProcrustesAlign:
    def test_output_shape(self):
        N, d, C = 8, 4, 6
        Z = torch.randn(N, d, C)
        anchor = torch.randn(d, C)
        out = batched_procrustes_align(Z, anchor)
        assert out.shape == (N, d, C)

    def test_no_nan(self):
        N, d, C = 8, 4, 6
        Z = torch.randn(N, d, C)
        anchor = torch.randn(d, C)
        out = batched_procrustes_align(Z, anchor)
        assert torch.isfinite(out).all()

    def test_identity_preserved_when_aligned(self):
        """If all Z_i already equal anchor (up to scale), alignment is a near-no-op."""
        d, C = 3, 3
        anchor = torch.eye(d, C)
        Z = anchor.unsqueeze(0).expand(5, -1, -1).clone()
        out = batched_procrustes_align(Z, anchor, proper=True)
        torch.testing.assert_close(out, Z, atol=1e-4, rtol=1e-4)

    def test_covariance_preserved(self):
        """Procrustes is an isometry: ||Z~_i||_F = ||Z_i||_F."""
        N, d, C = 10, 4, 5
        Z = torch.randn(N, d, C)
        anchor = torch.randn(d, C)
        out = batched_procrustes_align(Z, anchor)
        norms_in = Z.norm(dim=(1, 2))
        norms_out = out.norm(dim=(1, 2))
        torch.testing.assert_close(norms_in, norms_out, atol=1e-5, rtol=1e-5)

    def test_proper_rotation_both_modes_run(self):
        """Both proper=True and proper=False produce valid, finite outputs."""
        N, d, C = 20, 4, 4
        Z = torch.randn(N, d, C)
        anchor = torch.randn(d, C)
        out_proper = batched_procrustes_align(Z, anchor, proper=True)
        out_improper = batched_procrustes_align(Z, anchor, proper=False)
        assert out_proper.shape == (N, d, C)
        assert out_improper.shape == (N, d, C)
        assert torch.isfinite(out_proper).all()
        assert torch.isfinite(out_improper).all()

    def test_o_gauge_invariant_improper_false(self):
        """With proper=False: Z'=R Z -> align(Z')=align(Z) for all R in O(d) (exact)."""
        N, d, C = 6, 3, 4
        Z = torch.randn(N, d, C)
        anchor = torch.randn(d, C)
        rots = _random_orth(N, d)  # full O(d)
        z_transformed = rots @ Z
        out_orig = batched_procrustes_align(Z, anchor, proper=False)
        out_trans = batched_procrustes_align(z_transformed, anchor, proper=False)
        torch.testing.assert_close(out_orig, out_trans, atol=1e-4, rtol=1e-4)

    def test_so_gauge_invariant_proper_true(self):
        """With proper=True: Z'=R Z -> align(Z')=align(Z) for R in SO(d) (det=+1)."""
        N, d, C = 6, 3, 4
        Z = torch.randn(N, d, C)
        anchor = torch.randn(d, C)
        rots = _random_so(N, d)  # SO(d) only
        z_transformed = rots @ Z
        out_orig = batched_procrustes_align(Z, anchor, proper=True)
        out_trans = batched_procrustes_align(z_transformed, anchor, proper=True)
        torch.testing.assert_close(out_orig, out_trans, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# SheafPool: basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d,C", [(2, 4), (3, 6), (4, 8), (1, 4)])
class TestSheafPoolShape:
    def test_output_shape_3d(self, d, C):
        N, B = 12, 3
        pool = _make_pool(d, C)
        H = _make_embeddings(N, d, C)
        batch = torch.tensor([0, 1, 2]).repeat_interleave(4)
        out = pool(H, batch)
        assert out.shape == (B, C)

    def test_output_shape_2d(self, d, C):
        N, B = 12, 3
        pool = _make_pool(d, C)
        H_flat = _make_embeddings(N, d, C).reshape(N, d * C)
        batch = torch.tensor([0, 1, 2]).repeat_interleave(4)
        out = pool(H_flat, batch)
        assert out.shape == (B, C)

    def test_2d_and_3d_agree(self, d, C):
        N = 8
        pool = _make_pool(d, C)
        H = _make_embeddings(N, d, C)
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out_3d = pool(H, batch)
            out_2d = pool(H.reshape(N, d * C), batch)
        torch.testing.assert_close(out_3d, out_2d)

    def test_no_nan(self, d, C):
        N = 10
        pool = _make_pool(d, C)
        H = _make_embeddings(N, d, C)
        batch = torch.zeros(N, dtype=torch.long)
        out = pool(H, batch)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SheafPool: single-graph edge cases
# ---------------------------------------------------------------------------


class TestSheafPoolSingleGraph:
    def test_batch_none_alias(self):
        d, C, N = 3, 6, 8
        pool = _make_pool(d, C)
        H = _make_embeddings(N, d, C)
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out_none = pool(H, None)
            out_batch = pool(H, batch)
        torch.testing.assert_close(out_none, out_batch)

    def test_single_node(self):
        d, C = 4, 8
        pool = _make_pool(d, C)
        H = torch.randn(1, d, C)
        out = pool(H, None)
        assert out.shape == (1, C)
        assert torch.isfinite(out).all()

    def test_two_nodes(self):
        d, C = 3, 6
        pool = _make_pool(d, C)
        H = torch.randn(2, d, C)
        out = pool(H, None)
        assert out.shape == (1, C)


# ---------------------------------------------------------------------------
# SheafPool: permutation invariance
# ---------------------------------------------------------------------------


class TestPermutationInvariance:
    @pytest.mark.parametrize("d,C", [(3, 6), (4, 8)])
    def test_single_graph_permutation(self, d, C):
        """Permuting all nodes of a single graph gives the same output."""
        N = 10
        pool = _make_pool(d, C)
        H = torch.randn(N, d, C)
        perm = torch.randperm(N)
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out = pool(H, batch)
            out_perm = pool(H[perm], batch)
        torch.testing.assert_close(out, out_perm, atol=1e-5, rtol=1e-5)

    def test_multi_graph_permutation(self):
        """Permuting nodes within each graph (not across graphs) gives same output."""
        d, C = 3, 6
        N = 12
        pool = _make_pool(d, C)
        H = torch.randn(N, d, C)
        batch = torch.tensor([0, 1, 2]).repeat_interleave(4)
        nodes_per_graph = N // 3
        perm = torch.cat(
            [torch.randperm(nodes_per_graph) + g * nodes_per_graph for g in range(3)]
        )
        with torch.no_grad():
            out = pool(H, batch)
            out_perm = pool(H[perm], batch[perm])
        torch.testing.assert_close(out, out_perm, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# SheafPool: O(d) gauge invariance
# ---------------------------------------------------------------------------


class TestGaugeInvariance:
    @pytest.mark.parametrize("d,C", [(3, 6), (4, 8)])
    def test_so_gauge_invariant_proper(self, d, C):
        """Per-node proper rotations (SO(d), det=+1) leave the output unchanged.

        With proper_rot=True (default), the Procrustes step restricts to SO(d).
        For SO(d) gauge transforms the invariance is exact:
          S' = R S R^T  (eps I is preserved since R is orthogonal)
          Z' = R Z  (whitening commutes with conjugation by orth matrices)
          diag(Z'^T Z') = diag(Z^T Z)  (O(d)-invariant channel energies)
          Procrustes: C'=R C -> R*'=R R* ->
            Z~'=R*'^T Z'=R*^T Z=Z~  (cancels exactly)
        """
        N = 10
        pool = _make_pool(d, C, proper_rot=True)
        H = torch.randn(N, d, C)
        rots = _random_so(N, d)  # det = +1 for all, so proper_rot=True stays consistent
        h_gauged = rots @ H  # [N, d, d] @ [N, d, C] = [N, d, C]
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out = pool(H, batch)
            out_gauged = pool(h_gauged, batch)
        torch.testing.assert_close(out, out_gauged, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("d,C", [(3, 6), (4, 8)])
    def test_full_o_gauge_invariant_improper(self, d, C):
        """Full O(d) gauge invariance (including reflections) with proper_rot=False.

        With proper_rot=False the unconstrained Procrustes rotation R*=UV^T in O(d)
        cancels any per-node orthogonal gauge transform exactly, including improper
        rotations (det=-1).
        """
        N = 10
        pool = _make_pool(d, C, proper_rot=False)
        H = torch.randn(N, d, C)
        rots = _random_orth(N, d)  # full O(d) -- det may be +/-1
        h_gauged = rots @ H
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out = pool(H, batch)
            out_gauged = pool(h_gauged, batch)
        torch.testing.assert_close(out, out_gauged, atol=1e-4, rtol=1e-4)

    def test_shared_orthogonal_gauge(self):
        """Shared proper rotation on all nodes leaves output unchanged."""
        d, C, N = 4, 8, 12
        pool = _make_pool(d, C, proper_rot=True)
        H = torch.randn(N, d, C)
        R = _random_so(1, d)[0]  # [d, d], det = +1
        h_gauged = R[None] @ H  # [1, d, d] @ [N, d, C] -> [N, d, C] via broadcast
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out = pool(H, batch)
            out_gauged = pool(h_gauged, batch)
        torch.testing.assert_close(out, out_gauged, atol=1e-4, rtol=1e-4)

    def test_general_gl_gauge_approximate(self):
        """General GL(d) gauge invariance holds approximately for small eps_whiten.

        The eps_whiten regulariser breaks exact GL(d) invariance.  With very small eps,
        the error is proportional to eps and the non-orthogonality of G.
        """
        d, C, N = 3, 6, 8
        pool = _make_pool(d, C, eps_whiten=1e-6, eps_eigh=1e-8)
        torch.manual_seed(7)
        H = torch.randn(N, d, C)
        # Small perturbation from orthogonal: G = R + 0.01 * noise
        Rs = _random_orth(N, d)
        Gs = Rs + 0.01 * torch.randn(N, d, d)
        H_gauged = Gs @ H
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out = pool(H, batch)
            out_gauged = pool(H_gauged, batch)
        # Should be close (not exact) -- generous tolerance
        torch.testing.assert_close(out, out_gauged, atol=0.1, rtol=0.1)


# ---------------------------------------------------------------------------
# SheafPool: gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("d,C", [(3, 6), (4, 8)])
    def test_backward_does_not_error(self, d, C):
        N = 10
        pool = SheafPool(stalk_dim=d, channel_dim=C)
        H = torch.randn(N, d, C, requires_grad=True)
        batch = torch.zeros(N, dtype=torch.long)
        out = pool(H, batch)
        out.sum().backward()
        assert H.grad is not None

    def test_learnable_params_get_grad(self):
        d, C, N = 3, 6, 8
        pool = SheafPool(stalk_dim=d, channel_dim=C)
        H = torch.randn(N, d, C)
        batch = torch.zeros(N, dtype=torch.long)
        out = pool(H, batch)
        out.sum().backward()
        assert pool.anchor.grad is not None
        assert pool.token.grad is not None
        # Score MLP params
        for param in pool.score.parameters():
            assert param.grad is not None
        # Collect MLP params
        for param in pool.token_mlp.parameters():
            assert param.grad is not None

    def test_grad_not_nan(self):
        d, C, N = 4, 8, 12
        pool = SheafPool(stalk_dim=d, channel_dim=C)
        H = torch.randn(N, d, C, requires_grad=True)
        batch = torch.tensor([0, 1, 2]).repeat_interleave(4)
        out = pool(H, batch)
        out.sum().backward()
        assert H.grad is not None
        assert torch.isfinite(H.grad).all()


# ---------------------------------------------------------------------------
# SheafPool: multi-graph batching
# ---------------------------------------------------------------------------


class TestMultiGraphBatch:
    def test_uneven_graphs(self):
        """Different number of nodes per graph is handled correctly."""
        d, C = 3, 6
        pool = _make_pool(d, C)
        # 3 graphs: 2, 5, 3 nodes
        H = torch.randn(10, d, C)
        batch = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
        out = pool(H, batch)
        assert out.shape == (3, C)
        assert torch.isfinite(out).all()

    def test_single_node_graphs(self):
        """Each graph has exactly one node."""
        d, C = 3, 6
        pool = _make_pool(d, C)
        N = 4
        H = torch.randn(N, d, C)
        batch = torch.arange(N)  # graph i has exactly node i
        out = pool(H, batch)
        assert out.shape == (N, C)
        assert torch.isfinite(out).all()

    def test_many_graphs(self):
        d, C = 3, 6
        pool = _make_pool(d, C)
        B, N_per = 20, 5
        N = B * N_per
        H = torch.randn(N, d, C)
        batch = torch.arange(B).repeat_interleave(N_per)
        out = pool(H, batch)
        assert out.shape == (B, C)

    def test_batch_independence(self):
        """Output for graph g is independent of the features of other graphs."""
        d, C = 3, 6
        pool = _make_pool(d, C)
        N = 8
        H = torch.randn(N, d, C)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        with torch.no_grad():
            out_joint = pool(H, batch)
        # Compute graph 0 and graph 1 separately
        H0 = H[:4]
        H1 = H[4:]
        batch0 = torch.zeros(4, dtype=torch.long)
        batch1 = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            out0 = pool(H0, batch0)
            out1 = pool(H1, batch1)
        torch.testing.assert_close(out_joint[0], out0[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_joint[1], out1[0], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# SheafPool: configuration options
# ---------------------------------------------------------------------------


class TestSheafPoolConfig:
    def test_custom_hidden_dim(self):
        d, C = 3, 6
        pool = _make_pool(d, C, hidden_dim=16)
        H = torch.randn(8, d, C)
        out = pool(H, None)
        assert out.shape == (1, C)

    def test_improper_rot_flag(self):
        d, C, N = 3, 6, 8
        pool_proper = _make_pool(d, C, proper_rot=True)
        pool_improper = _make_pool(d, C, proper_rot=False)
        H = torch.randn(N, d, C)
        batch = torch.zeros(N, dtype=torch.long)
        with torch.no_grad():
            out_p = pool_proper(H, batch)
            out_i = pool_improper(H, batch)
        assert out_p.shape == out_i.shape == (1, C)
        assert torch.isfinite(out_p).all()
        assert torch.isfinite(out_i).all()

    def test_large_eps_whiten(self):
        """Large eps_whiten degrades to identity whitening; should still run."""
        d, C, N = 3, 6, 8
        pool = _make_pool(d, C, eps_whiten=1.0)
        H = torch.randn(N, d, C)
        out = pool(H, None)
        assert out.shape == (1, C)
        assert torch.isfinite(out).all()

    def test_train_mode_gradient(self):
        """Model is differentiable in train mode too."""
        d, C, N = 3, 6, 8
        pool = SheafPool(stalk_dim=d, channel_dim=C)
        pool.train()
        H = torch.randn(N, d, C, requires_grad=True)
        out = pool(H, None)
        out.sum().backward()
        assert H.grad is not None
        assert torch.isfinite(H.grad).all()


# ---------------------------------------------------------------------------
# Stage F equivalence: element-wise **2 == diag(T_hat^T T_hat)
# ---------------------------------------------------------------------------


class TestStageFEquivalence:
    """(updated**2).sum(dim=1) is equivalent to diag(updated.mT @ updated).

    For updated in R^{B x d x C}:
        diag(updated.mT @ updated)[b, c] = sum_j updated[b,j,c]^2
                                         = (updated**2).sum(dim=1)[b, c]

    The **2 form is O(d*C) vs O(d*C^2) for the full matrix product, so it is
    both faster and numerically identical.
    """

    @pytest.mark.parametrize("B,d,C", [(4, 3, 6), (1, 5, 8), (8, 2, 4)])
    def test_sq_sum_equals_diag_matmul(self, B, d, C):
        updated = torch.randn(B, d, C)
        # Fast path used in SheafPool stage F.
        fast = (updated**2).sum(dim=1)  # [B, C]
        # Equivalent full matrix product -- off-diagonal entries are wasted work.
        full = torch.diagonal(updated.mT @ updated, dim1=-2, dim2=-1)  # [B, C]
        torch.testing.assert_close(fast, full, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Stage D equivalence: message passing == scatter_add
# ---------------------------------------------------------------------------


class TestStageDEquivalence:
    """Stage D propagate() is equivalent to scatter_add on weighted aligned stalks.

    For each graph g the virtual super-node accumulates:
        P_g = sum_{i: batch[i]==g}  alpha_i * Z_i_aligned

    Both the MessagePassing route (propagate/message/aggregate) and a direct
    scatter_add must produce the same P_g -- this test verifies that claim.
    """

    def test_propagate_equals_scatter_add(self):
        from torch_geometric.utils import scatter

        d, C, N = 3, 6, 10
        pool = _make_pool(d, C)
        H = _make_embeddings(N, d, C)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        B = int(batch.max().item()) + 1

        with torch.no_grad():
            Z_aligned, weights = pool._prepare_node_features(H, batch)

            # Reference: direct scatter_add (O(N*d*C), no edge index needed).
            P_scatter = scatter(
                Z_aligned * weights[:, None, None],
                batch,
                dim=0,
                dim_size=B,
                reduce="sum",
            )  # [B, d, C]

            # SheafPool route: propagate() over a bipartite edge index.
            src = torch.arange(N)
            dst = batch + N
            edge_index = torch.stack([src, dst], dim=0)
            out = pool.propagate(  # ty: ignore[missing-argument]
                edge_index,  # ty: ignore[invalid-argument-type]
                size=(N, N + B),
                Z_aligned=Z_aligned,
                weights=weights,
            )
            P_mp = out[N:]  # [B, d, C]

        torch.testing.assert_close(P_scatter, P_mp, atol=1e-6, rtol=1e-6)
