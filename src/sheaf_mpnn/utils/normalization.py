# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import torch
from torch_geometric.utils import degree, scatter


# @torch.compile(dynamic=True)
def batched_sym_matrix_pow(
    matrices: torch.Tensor, p: float, eps: float = 1e-8
) -> torch.Tensor:
    r"""Power of symmetric positive semi-definite matrices via eigendecomposition.

    Uses ``torch.linalg.eigh``, which is faster and more numerically stable than
    SVD for symmetric matrices. ``eigh`` reads only the lower triangular part, so
    minor floating-point asymmetries in the upper triangle are ignored.

    Args:
        matrices: A batch of symmetric PSD matrices [batch, d, d].
        p: Power exponent (e.g. :math:`-0.5` for the inverse square root).
        eps: Minimum eigenvalue clamp for numerical stability.

    Returns:
        :math:`\text{matrices}^p`, shape [batch, d, d].
    """
    eigvals, eigvecs = torch.linalg.eigh(matrices)
    vals_pow = eigvals.clamp(min=eps).pow(p)
    return eigvecs * vals_pow.unsqueeze(-2) @ eigvecs.transpose(-1, -2)


class _SymMatrixPowSVD(torch.autograd.Function):
    """Symmetric matrix power with the reference SVD forward and a
    degeneracy-safe backward.

    Generic SVD autograd divides by s_i^2 - s_j^2 and returns NaN on repeated
    singular values (the reference jitters the input to dodge this, which
    still fails on exact float ties). For symmetric input the true derivative
    is the Daleckii-Krein kernel (g(s_i)-g(s_j))/(s_i-s_j), whose limit at
    s_i == s_j is the finite g'(s); this backward uses it directly.
    """

    @staticmethod
    def forward(ctx, matrices: torch.Tensor, p: float) -> torch.Tensor:
        # Bit-identical for symmetric input; makes the Daleckii-Krein backward
        # the exact derivative of this node for any input.
        matrices = 0.5 * (matrices + matrices.transpose(-1, -2))
        vecs, vals, _ = torch.linalg.svd(matrices)
        good = (
            vals
            > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
        )
        zero = torch.zeros((), device=matrices.device, dtype=matrices.dtype)
        gvals = torch.where(good, vals.pow(p), zero)
        ctx.save_for_backward(vecs, vals, gvals, good)
        ctx.p = p
        return (vecs * gvals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # ty: ignore[invalid-method-override]
        vecs, vals, gvals, good = ctx.saved_tensors
        p: float = ctx.p
        zero = torch.zeros((), device=vals.device, dtype=vals.dtype)
        dvals = torch.where(good, p * vals.pow(p - 1), zero)

        diff = vals.unsqueeze(-1) - vals.unsqueeze(-2)
        gdiff = gvals.unsqueeze(-1) - gvals.unsqueeze(-2)
        # Within float cancellation range of a tie, the divided difference is
        # numerically meaningless; switch to its analytic limit g'(s).
        tol = vals.max(-1, True).values.unsqueeze(-1) * (
            10 * torch.finfo(vals.dtype).eps
        )
        near = diff.abs() <= tol
        kernel = torch.where(
            near,
            0.5 * (dvals.unsqueeze(-1) + dvals.unsqueeze(-2)),
            gdiff / torch.where(near, torch.ones_like(diff), diff),
        )
        # Upstream graphs only ever feed symmetric matrices here, so the
        # antisymmetric part of grad_out cannot contribute.
        sym_grad = 0.5 * (grad_out + grad_out.transpose(-1, -2))
        inner = torch.transpose(vecs, -2, -1) @ sym_grad @ vecs
        grad_matrices = vecs @ (kernel * inner) @ torch.transpose(vecs, -2, -1)
        return grad_matrices, None


def batched_sym_matrix_pow_svd(matrices: torch.Tensor, p: float) -> torch.Tensor:
    """Power of symmetric matrices using SVD (reference forward) with a
    backward that stays finite on repeated singular values.

    Args:
        matrices: A batch of symmetric PSD matrices [batch, d, d]
        p: Power exponent

    Returns:
        Power of each matrix: matrices^p
    """
    return _SymMatrixPowSVD.apply(matrices, p)


# NOTE: augmented normalization (Bodnar et al.): invert D+1 / D+I, never the
# raw sheaf degree D. Callers must not also add self-loops (double augmentation).
def apply_diagonal_norm(
    self_map: torch.Tensor,
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise (D+1)^{-1/2} normalization for diagonal restriction maps.

    For F = diag(w), F^T F = diag(w^2). The sheaf degree D_v = Sigma w^2 is a
    d-vector.
    """
    dst, src = edge_index[1], edge_index[0]
    degree_vec = scatter(self_map, dst, dim=0, dim_size=num_nodes, reduce="sum")
    degree_sqrt_inv = (degree_vec + 1).pow(-0.5)

    norm_dst = degree_sqrt_inv[dst]
    norm_src = degree_sqrt_inv[src]
    return norm_dst * self_map * norm_dst, norm_dst * cross_map * norm_src


def apply_orthogonal_norm(
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    self_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric augmented degree normalization for orthogonal restriction maps.

    Because F^T F = I, D_v = deg(v) * I so (D+I)^{-1/2} = (deg(v)+1)^{-1/2} * I.
    The +1 augmentation replaces the former self-loop-based degree inflation
    (add_self_loops now defaults to False) and keeps isolated nodes finite.

    With ``self_diag`` (per-edge, per-channel diagonal contributions, e.g.
    ``w_e^2`` on learned channels and 1 on fixed lp/hp channels for the
    edge-weighted bundle), the degree is the weighted per-channel scatter and
    the normalization is applied channelwise, as in the reference builder.

    Returns:
        Tuple of (norm_self, norm_cross): norm_self is the per-edge diagonal
        coefficient (``[E, 1, 1]`` scalar, or ``[E, d, 1]`` channelwise when
        ``self_diag`` is given); norm_cross is the cross-map with the
        ``(D+1)^{-1/2}`` scaling on both sides already applied.
    """
    dst, src = edge_index[1], edge_index[0]
    if self_diag is None:
        deg = degree(dst, num_nodes=num_nodes, dtype=cross_map.dtype)
        deg_sqrt_inv = (deg + 1).pow(-0.5)
        norm_self = (deg_sqrt_inv[dst] ** 2).view(-1, 1, 1)
        norm_cross = (deg_sqrt_inv[dst] * deg_sqrt_inv[src]).view(-1, 1, 1)
        return norm_self, norm_cross * cross_map

    deg_vec = scatter(self_diag, dst, dim=0, dim_size=num_nodes, reduce="sum")
    dinv = (deg_vec + 1).pow(-0.5)
    norm_self = (self_diag * dinv[dst] ** 2).unsqueeze(-1)
    norm_cross = dinv[dst].unsqueeze(-1) * cross_map * dinv[src].unsqueeze(-2)
    return norm_self, norm_cross


def _block_degree_sqrt_inv(
    degree_mat: torch.Tensor,
    stalk_dim: int,
    learned_dim: int | None,
    training: bool,
) -> torch.Tensor:
    """(D+I)^{-1/2} with fixed lp/hp channels kept OUT of the SVD graph.

    The SVD backward divides by differences of singular values; fixed channels
    sit at exactly deg+1 and collide with learned ones during training, so
    (as in the reference) they are normalized in closed form instead.
    """
    d = stalk_dim if learned_dim is None else learned_dim
    num_nodes = degree_mat.size(0)

    # Train-time jitter perturbs the O(1) identity augmentation, never the raw
    # degree, so SVD gradients stay finite while train/eval operators coincide.
    eps = torch.zeros(d, device=degree_mat.device, dtype=degree_mat.dtype)
    if training:
        eps = eps.uniform_(-0.001, 0.001)
    learned_deg = degree_mat[:, :d, :d] + torch.diag_embed(1.0 + eps)
    inv_learned = batched_sym_matrix_pow_svd(learned_deg, -0.5)
    if d == stalk_dim:
        return inv_learned

    # Fixed channels have diagonal degree deg(v): closed-form (deg+1)^{-1/2}.
    fixed_deg = degree_mat[:, d:, d:].diagonal(dim1=-2, dim2=-1)
    inv_fixed = (fixed_deg + 1.0).pow(-0.5)
    full = degree_mat.new_zeros(num_nodes, stalk_dim, stalk_dim)
    full[:, :d, :d] = inv_learned
    full[:, d:, d:] = torch.diag_embed(inv_fixed)
    return full


def apply_low_rank_norm(
    self_map: torch.Tensor,
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    stalk_dim: int,
    training: bool,
    learned_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric (D+I)^{-1/2} normalization for low_rank restriction maps.

    Builds the sheaf degree D_v = Sigma F^T F by scatter-adding per-edge
    self_map products, then computes (D+I)^{-1/2} via SVD for stability.
    ``learned_dim`` marks the leading learned block; trailing fixed lp/hp
    channels are normalized in closed form outside the SVD.
    """
    dst, src = edge_index[1], edge_index[0]
    degree_mat = scatter(self_map, dst, dim=0, dim_size=num_nodes, reduce="sum")

    degree_mat_sqrt_inv = _block_degree_sqrt_inv(
        degree_mat, stalk_dim, learned_dim, training
    )
    degree_inv_dst = degree_mat_sqrt_inv[dst]
    degree_inv_src = degree_mat_sqrt_inv[src]
    norm_self = degree_inv_dst @ self_map @ degree_inv_dst
    norm_cross = degree_inv_dst @ cross_map @ degree_inv_src

    if not torch.all(torch.isfinite(norm_self)):
        raise RuntimeError("norm_self contains non-finite values after normalization")
    if not torch.all(torch.isfinite(norm_cross)):
        raise RuntimeError("norm_cross contains non-finite values after normalization")

    return norm_self, norm_cross


def apply_general_norm(
    self_map: torch.Tensor,
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    stalk_dim: int,
    training: bool,
    learned_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric (D+I)^{-1/2} normalization for general (full-matrix) maps.

    Builds the sheaf degree D_v = Sigma F^T F by scatter-adding per-edge
    self_map products, then computes (D+I)^{-1/2} via SVD (reference choice:
    more robust than eigh on ill-conditioned degree blocks). ``learned_dim``
    marks the leading learned block; trailing fixed lp/hp channels are
    normalized in closed form outside the SVD.
    """
    dst, src = edge_index[1], edge_index[0]
    degree_mat = scatter(self_map, dst, dim=0, dim_size=num_nodes, reduce="sum")

    degree_mat_sqrt_inv = _block_degree_sqrt_inv(
        degree_mat, stalk_dim, learned_dim, training
    )
    degree_inv_dst = degree_mat_sqrt_inv[dst]
    degree_inv_src = degree_mat_sqrt_inv[src]
    # Clamp to [-1, 1] as in the reference to bound the operator's blocks.
    norm_self = (degree_inv_dst @ self_map @ degree_inv_dst).clamp(min=-1, max=1)
    norm_cross = (degree_inv_dst @ cross_map @ degree_inv_src).clamp(min=-1, max=1)

    if not torch.all(torch.isfinite(norm_self)):
        raise RuntimeError("norm_self contains non-finite values after normalization")
    if not torch.all(torch.isfinite(norm_cross)):
        raise RuntimeError("norm_cross contains non-finite values after normalization")

    return norm_self, norm_cross


# Re-export for callers that reference this module directly via monkeypatching.
__all__ = [
    "batched_sym_matrix_pow",
    "batched_sym_matrix_pow_svd",
    "apply_diagonal_norm",
    "apply_orthogonal_norm",
    "apply_low_rank_norm",
    "apply_general_norm",
]
