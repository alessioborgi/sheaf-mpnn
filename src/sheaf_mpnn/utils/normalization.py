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
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise (D+1)^{-1/2} normalization for diagonal restriction maps.

    For F = diag(w), F^T F = diag(w^2). The sheaf degree D_v = Sigma w^2 is a
    d-vector.

    ``detach_degree`` stops gradients from flowing back through the degree
    vector into the restriction maps that produced it.
    """
    dst, src = edge_index[1], edge_index[0]
    degree_vec = scatter(self_map, dst, dim=0, dim_size=num_nodes, reduce="sum")
    if detach_degree:
        degree_vec = degree_vec.detach()
    degree_sqrt_inv = (degree_vec + 1).pow(-0.5)

    norm_dst = degree_sqrt_inv[dst]
    norm_src = degree_sqrt_inv[src]
    return norm_dst * self_map * norm_dst, norm_dst * cross_map * norm_src


def apply_orthogonal_norm(
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    self_diag: torch.Tensor | None = None,
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric augmented degree normalization for orthogonal restriction maps.

    Because F^T F = I, D_v = deg(v) * I so (D+I)^{-1/2} = (deg(v)+1)^{-1/2} * I.
    The +1 augmentation replaces the former self-loop-based degree inflation
    (add_self_loops now defaults to False) and keeps isolated nodes finite.

    With ``self_diag`` (per-edge, per-channel diagonal contributions, e.g.
    ``w_e^2`` on learned channels and 1 on fixed lp/hp channels for the
    edge-weighted bundle), the degree is the weighted per-channel scatter and
    the normalization is applied channelwise, as in the reference builder.

    ``detach_degree`` stops gradients from flowing back through the degree
    vector into the restriction maps that produced it. It only affects the
    ``self_diag`` path, since the plain-degree path uses the graph's
    (non-differentiable) node degree already.

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
    if detach_degree:
        deg_vec = deg_vec.detach()
    dinv = (deg_vec + 1).pow(-0.5)
    norm_self = (self_diag * dinv[dst] ** 2).unsqueeze(-1)
    norm_cross = dinv[dst].unsqueeze(-1) * cross_map * dinv[src].unsqueeze(-2)
    return norm_self, norm_cross


def apply_diagonal_coboundary_norm(
    map_dst: torch.Tensor,
    map_src: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise (D+1)^{-1/2} normalization of diagonal restriction maps.

    For F = diag(w) the sheaf degree D_v = Sigma w^2 is a d-vector, exactly as in
    ``apply_diagonal_norm``; here it scales each endpoint map instead of their
    products.

    Args:
        map_dst: Diagonal entries of F at the destination endpoint [E, d].
        map_src: Diagonal entries of F at the source endpoint [E, d].
        edge_index: Graph connectivity [2, E] holding both edge orientations.
        num_nodes: Number of nodes in the graph.

    Returns:
        ``(F_dst D_dst^{-1/2}, F_src D_src^{-1/2})``, both [E, d].
    """
    dst, src = edge_index[1], edge_index[0]
    degree_vec = scatter(map_dst**2, dst, dim=0, dim_size=num_nodes, reduce="sum")
    degree_sqrt_inv = (degree_vec + 1).pow(-0.5)
    return degree_sqrt_inv[dst] * map_dst, degree_sqrt_inv[src] * map_src


def apply_orthogonal_coboundary_norm(
    map_dst: torch.Tensor,
    map_src: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric (deg+1)^{-1/2} normalization of orthogonal restriction maps.

    Because F^T F = I the sheaf degree is ``deg(v) * I`` and the block inverse
    square root collapses to a scalar, mirroring ``apply_orthogonal_norm``.
    With ``edge_weight`` (the reference EdgeWeightLearner scalar) the maps are
    rescaled by w_e and the degree becomes the weighted ``Sigma w_e^2``.

    Args:
        map_dst: Orthogonal F at the destination endpoint [E, d, d].
        map_src: Orthogonal F at the source endpoint [E, d, d].
        edge_index: Graph connectivity [2, E] holding both edge orientations.
        num_nodes: Number of nodes in the graph.
        edge_weight: Optional symmetric per-edge scalar [E, 1].

    Returns:
        ``(F_dst D_dst^{-1/2}, F_src D_src^{-1/2})``, both [E, d, d].
    """
    dst, src = edge_index[1], edge_index[0]
    if edge_weight is None:
        deg = degree(dst, num_nodes=num_nodes, dtype=map_dst.dtype)
    else:
        deg = scatter(
            edge_weight**2, dst, dim=0, dim_size=num_nodes, reduce="sum"
        ).squeeze(-1)
        weight = edge_weight.unsqueeze(-1)
        map_dst, map_src = map_dst * weight, map_src * weight
    deg_sqrt_inv = (deg + 1).pow(-0.5)
    return (
        deg_sqrt_inv[dst].view(-1, 1, 1) * map_dst,
        deg_sqrt_inv[src].view(-1, 1, 1) * map_src,
    )


def apply_general_coboundary_norm(
    map_dst: torch.Tensor,
    map_src: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    stalk_dim: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Symmetric (D+I)^{-1/2} normalisation of full-matrix restriction maps.

    Serves both the general and the low-rank families: a rank-r map is a full
    ``d x d`` matrix once its factors are multiplied out, and the sheaf degree
    ``D_v = Sigma_e F_{v<|e}^T F_{v<|e}`` is a ``d x d`` PSD block either way.
    Its inverse square root therefore multiplies each endpoint map on the
    *right*, which is what ``delta D^{-1/2}`` / ``D^{-1/2} delta^T`` (Eq. 13-14)
    asks for and what collapses to ``apply_general_norm`` when Phi = identity.

    Two differences from the linear ``apply_general_norm``:

    * no ``[-1, 1]`` clamp -- the reference clamps the assembled *products*,
      which has no equivalent at the level of a single endpoint map;
    * the ``+I`` augmentation is what keeps this defined for low-rank maps,
      whose degree block is singular whenever ``rank * deg(v) < d``.

    Because ``edge_index`` carries both orientations, a single scatter over
    ``dst`` already accumulates every node's full degree, so ``D_src`` is the
    same table read at ``src``.

    Args:
        map_dst: F at the destination endpoint [E, d, d].
        map_src: F at the source endpoint [E, d, d].
        edge_index: Graph connectivity [2, E] holding both edge orientations.
        num_nodes: Number of nodes in the graph.
        stalk_dim: Stalk dimension d.
        training: Whether to jitter the identity augmentation for SVD stability.

    Returns:
        ``(F_dst D_dst^{-1/2}, F_src D_src^{-1/2})``, both [E, d, d].
    """
    dst, src = edge_index[1], edge_index[0]
    self_prod = torch.matmul(map_dst.transpose(-2, -1), map_dst)
    degree_mat = scatter(self_prod, dst, dim=0, dim_size=num_nodes, reduce="sum")
    # learned_dim=None: NLSD carries no fixed lp/hp channels to keep out of the SVD.
    degree_mat_sqrt_inv = _block_degree_sqrt_inv(degree_mat, stalk_dim, None, training)

    norm_dst = torch.matmul(map_dst, degree_mat_sqrt_inv[dst])
    norm_src = torch.matmul(map_src, degree_mat_sqrt_inv[src])

    if not torch.all(torch.isfinite(norm_dst)):
        raise RuntimeError("norm_dst contains non-finite values after normalization")
    if not torch.all(torch.isfinite(norm_src)):
        raise RuntimeError("norm_src contains non-finite values after normalization")

    return norm_dst, norm_src


def _block_degree_sqrt_inv(
    degree_mat: torch.Tensor,
    stalk_dim: int,
    learned_dim: int | None,
    training: bool,
    detach_degree: bool = False,
) -> torch.Tensor:
    """(D+I)^{-1/2} with fixed lp/hp channels kept OUT of the SVD graph.

    The SVD backward divides by differences of singular values; fixed channels
    sit at exactly deg+1 and collide with learned ones during training, so
    (as in the reference) they are normalized in closed form instead.

    ``detach_degree`` stops gradients from flowing back through the degree
    matrix into the restriction maps that produced it by detaching the degree
    blocks before computing the inverse square root. Downstream ops remain
    differentiable w.r.t. the restriction maps, but not through the degree term.
    """
    d = stalk_dim if learned_dim is None else learned_dim
    num_nodes = degree_mat.size(0)

    # Train-time jitter perturbs the O(1) identity augmentation, never the raw
    # degree, so SVD gradients stay finite while train/eval operators coincide.
    eps = torch.zeros(d, device=degree_mat.device, dtype=degree_mat.dtype)
    if training:
        eps = eps.uniform_(-0.001, 0.001)
    if detach_degree:
        learned_deg = degree_mat[:, :d, :d].detach() + torch.diag_embed(1.0 + eps)
    else:
        learned_deg = degree_mat[:, :d, :d] + torch.diag_embed(1.0 + eps)
    inv_learned = batched_sym_matrix_pow_svd(learned_deg, -0.5)
    if d == stalk_dim:
        return inv_learned

    # Fixed channels have diagonal degree deg(v): closed-form (deg+1)^{-1/2}.
    fixed_deg = degree_mat[:, d:, d:].diagonal(dim1=-2, dim2=-1)
    if detach_degree:
        fixed_deg = fixed_deg.detach()
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
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric (D+I)^{-1/2} normalization for low_rank restriction maps.

    Builds the sheaf degree D_v = Sigma F^T F by scatter-adding per-edge
    self_map products, then computes (D+I)^{-1/2} via SVD for stability.
    ``learned_dim`` marks the leading learned block; trailing fixed lp/hp
    channels are normalized in closed form outside the SVD. ``detach_degree``
    is forwarded to :func:`_block_degree_sqrt_inv` to stop gradients from
    flowing back through the degree matrix into the restriction maps.
    """
    dst, src = edge_index[1], edge_index[0]
    degree_mat = scatter(self_map, dst, dim=0, dim_size=num_nodes, reduce="sum")

    degree_mat_sqrt_inv = _block_degree_sqrt_inv(
        degree_mat, stalk_dim, learned_dim, training, detach_degree=detach_degree
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
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric (D+I)^{-1/2} normalization for general (full-matrix) maps.

    Builds the sheaf degree D_v = Sigma F^T F by scatter-adding per-edge
    self_map products, then computes (D+I)^{-1/2} via SVD (reference choice:
    more robust than eigh on ill-conditioned degree blocks). ``learned_dim``
    marks the leading learned block; trailing fixed lp/hp channels are
    normalized in closed form outside the SVD. ``detach_degree`` is forwarded
    to :func:`_block_degree_sqrt_inv` to stop gradients from flowing back
    through the degree matrix into the restriction maps.
    """
    dst, src = edge_index[1], edge_index[0]
    degree_mat = scatter(self_map, dst, dim=0, dim_size=num_nodes, reduce="sum")

    degree_mat_sqrt_inv = _block_degree_sqrt_inv(
        degree_mat, stalk_dim, learned_dim, training, detach_degree=detach_degree
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


# ---------------------------------------------------------------------------
# Directed normalisation -- edge-index convention
#
# PyG stores edges as edge_index = [[u, ...], [v, ...]], i.e.
#   src = edge_index[0]  ->  the tail node  (signal flows *from* src)
#   dst = edge_index[1]  ->  the head node  (signal flows *into* dst)
#
# Two asymmetric degree matrices are built:
#   D_out[u] = sum_{e: src(e)=u}  F_{ue}^T F_{ue}   (out-degree block at src)
#   D_in[v]  = sum_{e: dst(e)=v}  F_{ve}^T F_{ve}   (in-degree block at dst)
#
# The forward map  (u -> v) is normalised by D_in[dst] on the left and
# D_out[src] on the right, matching the directed Laplacian  Delta = B^T Delta_e B.
# The backward map (v -> u) transposes both the restriction map and the
# degree-matrix roles, which preserves the adjoint structure.
# ---------------------------------------------------------------------------
def apply_directed_diagonal_norm(
    dst_map: torch.Tensor,
    src_map: torch.Tensor,
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed D^{-1/2} normalisation for diagonal restriction maps.

    For F = diag(w), F^T F = diag(w^2), so all operations are elementwise.

    dst_map   = w_{ve}^2  [E, d]  target-side self-products
    src_map   = w_{ue}^2  [E, d]  source-side self-products
    cross_map = w_{ve} * w_{ue}  [E, d]

    D_in[v]  = scatter(dst_map, dst)
    D_out[u] = scatter(src_map, src)

    norm_fwd[e] = D_in[dst]^{-1/2}  * cross_map * D_out[src]^{-1/2}
    norm_bwd[e] = D_out[src]^{-1/2} * cross_map * D_in[dst]^{-1/2}

    ``detach_degree`` stops gradients from flowing back through D_in/D_out
    into the restriction maps that produced them.
    """
    src, dst = edge_index[0], edge_index[1]

    D_in = scatter(dst_map, dst, dim=0, dim_size=num_nodes, reduce="sum")
    D_out = scatter(src_map, src, dim=0, dim_size=num_nodes, reduce="sum")
    if detach_degree:
        D_in = D_in.detach()
        D_out = D_out.detach()
    D_in_inv_sqrt = D_in.clamp(min=1e-8).pow(-0.5)
    D_out_inv_sqrt = D_out.clamp(min=1e-8).pow(-0.5)

    norm_fwd = D_in_inv_sqrt[dst] * cross_map * D_out_inv_sqrt[src]
    norm_bwd = D_out_inv_sqrt[src] * cross_map * D_in_inv_sqrt[dst]

    return norm_fwd, norm_bwd


def apply_directed_orthogonal_norm(
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    detach_degree: bool = False,  # noqa: ARG001 - kept for interface parity
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed D^{-1/2} normalisation for orthogonal restriction maps.

    F^T F = I, so D_in[v] = in_deg(v) * I and D_out[u] = out_deg(u) * I.
    The block inverse square root reduces to a scalar prefactor.

    norm_fwd[e] = in_deg[dst]^{-1/2}  * cross_map   * out_deg[src]^{-1/2}
    norm_bwd[e] = out_deg[src]^{-1/2} * cross_map^T * in_deg[dst]^{-1/2}

    ``detach_degree`` is accepted for signature parity with the other
    ``apply_*_norm`` functions but has no effect: the in/out degrees here
    come from the graph's node-degree counts (via ``degree()``), which carry
    no gradient in the first place.
    """
    src, dst = edge_index[0], edge_index[1]

    in_deg = degree(dst, num_nodes=num_nodes, dtype=cross_map.dtype)
    out_deg = degree(src, num_nodes=num_nodes, dtype=cross_map.dtype)
    in_deg_inv_sqrt = in_deg.clamp(min=1e-8).pow(-0.5)
    out_deg_inv_sqrt = out_deg.clamp(min=1e-8).pow(-0.5)

    scalar = (in_deg_inv_sqrt[dst] * out_deg_inv_sqrt[src]).view(-1, 1, 1)

    norm_fwd = scalar * cross_map
    norm_bwd = scalar * cross_map.transpose(-2, -1)

    return norm_fwd, norm_bwd


def apply_directed_general_norm(
    dst_map: torch.Tensor,
    src_map: torch.Tensor,
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    stalk_dim: int,
    training: bool,
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed D^{-1/2} normalisation for general restriction maps.

    dst_map   = F_{ve}^T F_{ve}  [E, d, d]  target-side self-products
    src_map   = F_{ue}^T F_{ue}  [E, d, d]  source-side self-products
    cross_map = F_{ve}^T F_{ue}  [E, d, d]

    D_in[v]  = scatter(dst_map, dst)
    D_out[u] = scatter(src_map, src)

    norm_fwd[e] = D_in[dst]^{-1/2}  @ cross_map   @ D_out[src]^{-1/2}
    norm_bwd[e] = D_out[src]^{-1/2} @ cross_map^T @ D_in[dst]^{-1/2}

    Uses eigh-based matrix power. Suitable when degree matrices are well-conditioned.
    ``detach_degree`` stops gradients from flowing back through D_in/D_out
    into the restriction maps that produced them.
    """
    src, dst = edge_index[0], edge_index[1]

    eps = (
        torch.empty(stalk_dim, device=dst_map.device, dtype=dst_map.dtype).uniform_(
            -0.001, 0.001
        )
        if training
        else torch.zeros(stalk_dim, device=dst_map.device, dtype=dst_map.dtype)
    )
    eps_diag = torch.diag_embed(eps)

    D_in = scatter(dst_map, dst, dim=0, dim_size=num_nodes, reduce="sum") + eps_diag
    D_out = scatter(src_map, src, dim=0, dim_size=num_nodes, reduce="sum") + eps_diag
    if detach_degree:
        D_in = D_in.detach()
        D_out = D_out.detach()
    D_in_inv_sqrt = batched_sym_matrix_pow(D_in, -0.5)
    D_out_inv_sqrt = batched_sym_matrix_pow(D_out, -0.5)

    norm_fwd = D_in_inv_sqrt[dst] @ cross_map @ D_out_inv_sqrt[src]
    norm_bwd = D_out_inv_sqrt[src] @ cross_map.transpose(-2, -1) @ D_in_inv_sqrt[dst]

    if not torch.all(torch.isfinite(norm_fwd)):
        raise RuntimeError("norm_fwd contains non-finite values after normalisation")
    if not torch.all(torch.isfinite(norm_bwd)):
        raise RuntimeError("norm_bwd contains non-finite values after normalisation")

    return norm_fwd, norm_bwd


def apply_directed_low_rank_norm(
    dst_map: torch.Tensor,
    src_map: torch.Tensor,
    cross_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    stalk_dim: int,
    training: bool,
    detach_degree: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed D^{-1/2} normalisation for low-rank restriction maps.

    dst_map   = F_{ve}^T F_{ve}  [E, d, d]  target-side self-products
    src_map   = F_{ue}^T F_{ue}  [E, d, d]  source-side self-products
    cross_map = F_{ve}^T F_{ue}  [E, d, d]

    D_in[v]  = scatter(dst_map, dst)
    D_out[u] = scatter(src_map, src)

    norm_fwd[e] = D_in[dst]^{-1/2}  @ cross_map   @ D_out[src]^{-1/2}
    norm_bwd[e] = D_out[src]^{-1/2} @ cross_map^T @ D_in[dst]^{-1/2}

    Uses SVD-based matrix power for stability when degree matrices are rank-deficient.
    ``detach_degree`` stops gradients from flowing back through D_in/D_out
    into the restriction maps that produced the; defaults to false
    """
    src, dst = edge_index[0], edge_index[1]

    eps = (
        torch.empty(stalk_dim, device=dst_map.device, dtype=dst_map.dtype).uniform_(
            -0.001, 0.001
        )
        if training
        else torch.zeros(stalk_dim, device=dst_map.device, dtype=dst_map.dtype)
    )
    eps_diag = torch.diag_embed(eps)

    D_in = scatter(dst_map, dst, dim=0, dim_size=num_nodes, reduce="sum") + eps_diag
    D_out = scatter(src_map, src, dim=0, dim_size=num_nodes, reduce="sum") + eps_diag
    if detach_degree:
        D_in = D_in.detach()
        D_out = D_out.detach()
    D_in_inv_sqrt = batched_sym_matrix_pow_svd(D_in, -0.5)
    D_out_inv_sqrt = batched_sym_matrix_pow_svd(D_out, -0.5)

    norm_fwd = D_in_inv_sqrt[dst] @ cross_map @ D_out_inv_sqrt[src]
    norm_bwd = D_out_inv_sqrt[src] @ cross_map.transpose(-2, -1) @ D_in_inv_sqrt[dst]

    if not torch.all(torch.isfinite(norm_fwd)):
        raise RuntimeError("norm_fwd contains non-finite values after normalisation")
    if not torch.all(torch.isfinite(norm_bwd)):
        raise RuntimeError("norm_bwd contains non-finite values after normalisation")

    return norm_fwd, norm_bwd


# Re-export for callers that reference this module directly via monkeypatching.
__all__ = [
    "batched_sym_matrix_pow",
    "batched_sym_matrix_pow_svd",
    "apply_diagonal_norm",
    "apply_orthogonal_norm",
    "apply_low_rank_norm",
    "apply_general_norm",
    "apply_diagonal_coboundary_norm",
    "apply_orthogonal_coboundary_norm",
    "apply_general_coboundary_norm",
    "apply_directed_diagonal_norm",
    "apply_directed_orthogonal_norm",
    "apply_directed_general_norm",
    "apply_directed_low_rank_norm",
]
