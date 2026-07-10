# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import functools
import math
import warnings

import numpy as np
import torch


def cayley(params: torch.Tensor, d: int, clamp_val: float = 10.0) -> torch.Tensor:
    """Cayley transform from skew-symmetric matrix entries to orthogonal W.

    Args:
        params: Tensor of shape ``[N, d*(d-1)//2]`` containing the independent upper
            triangular entries of the skew-symmetric matrix.
        d: The dimension of the resulting orthogonal matrix.
        clamp_val: Maximum absolute value for clamping parameters before constructing
            the skew-symmetric matrix to avoid ill-conditioned solves.

    Returns:
        Orthogonal restriction map of shape ``[N, d, d]``.
    """
    params = torch.clamp(params, -clamp_val, clamp_val)
    A = torch.zeros(params.size(0), d, d, device=params.device, dtype=params.dtype)
    indices = torch.triu_indices(d, d, offset=1)

    A[:, indices[0], indices[1]] = params
    A = A - A.transpose(-2, -1)

    # Half-Cayley (I - A/2)^{-1}(I + A/2) as in the reference Orthogonal
    # transform; halving keeps rotation angles moderate for bounded params.
    eye = torch.eye(d, device=params.device, dtype=params.dtype).unsqueeze(0)
    return torch.linalg.solve(eye - A / 2, eye + A / 2)


def attention_cayley(
    raw: torch.Tensor, d: int, device=None, dtype=None
) -> torch.Tensor:
    """Cayley transform via skew-symmetric part of I - softmax(raw).

    Args:
        raw: Raw un-normalized attention parameters of shape ``[N, d * d]``.
        d: The dimension of the resulting orthogonal matrix.
        device: Target tensor device.
        dtype: Target tensor dtype.

    Returns:
        Orthogonal restriction map of shape ``[N, d, d]``.
    """
    if device is None:
        device = raw.device
    if dtype is None:
        dtype = raw.dtype
    eye = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    M = eye - torch.softmax(raw.view(-1, d, d), dim=-1)
    A = (M - M.transpose(-2, -1)) / 2
    return torch.linalg.solve(eye - A, eye + A)


def householder(
    params: torch.Tensor, d: int, stop_recursion: int | None = None
) -> torch.Tensor:
    """Batched, autograd-safe FastH++ implementation (free reflectors).
    from: github.com/alexandermath/fasth (fasthpp.py)
    What if Neural Networks had SVDs? NeurIPS 2020.
    Converts ``[batch_size, d*d]`` params (d free reflector vectors) to
    orthogonal ``[batch_size, d, d]`` via the parallel WY merge.

    Args:
        params: Raw Householder parameters of shape ``[batch_size, d * d]``.
        d: The dimension of the resulting orthogonal matrix.
        stop_recursion: If provided, stop the parallel reduction at this step.

    Returns:
        Orthogonal restriction map of shape ``[batch_size, d, d]``.
    """
    batch_size, _ = params.shape

    V = params.view(batch_size, d, d)
    norms = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    V = V / norms

    return _reflection_product(V, d, stop_recursion)


def _compact_reflectors(params: torch.Tensor, d: int) -> torch.Tensor:
    """LAPACK-style compact reflectors: row i has a unit pivot at position i
    and the free parameters strictly after it (d*(d-1)//2 params total).
    """
    batch_size = params.size(0)
    V = torch.zeros(batch_size, d, d, device=params.device, dtype=params.dtype)
    # Transpose of the reference's tril scatter, so V rows equal the orgqr
    # column reflectors parameter-for-parameter.
    idx = torch.tril_indices(d, d, offset=-1, device=params.device)
    V[:, idx[1], idx[0]] = params
    V = V + torch.eye(d, device=params.device, dtype=params.dtype)
    return V / V.norm(dim=-1, keepdim=True)


def fasthpp(params: torch.Tensor, d: int) -> torch.Tensor:
    """FastH++ on the compact householder parameterization.

    Computes the same orthogonal map as ``householder_orgqr`` (the reference
    parameterization of Bodnar et al.) using only native torch ops: the
    product of d reflections whose vectors have a unit pivot and
    ``d*(d-1)//2`` free entries.

    Args:
        params: Compact parameters of shape ``[batch_size, d*(d-1)//2]``.
        d: The dimension of the resulting orthogonal matrix.

    Returns:
        Orthogonal restriction map of shape ``[batch_size, d, d]``.
    """
    return _reflection_product(_compact_reflectors(params, d), d)


@functools.lru_cache(maxsize=1)
def _load_torch_householder_orgqr():
    """Import torch_householder once, warning a single time if unavailable.

    The C++ extension can fail to load (ABI mismatch, missing ninja); fasthpp
    computes the identical map with native torch ops. Cached so the fallback
    warning fires once rather than on every forward call. Returns the orgqr
    callable, or None when the extension is unavailable.
    """
    try:
        from torch_householder import (  # ty: ignore[unresolved-import]
            torch_householder_orgqr,
        )

        return torch_householder_orgqr
    except (ImportError, OSError, RuntimeError):  # pragma: no cover
        warnings.warn(
            "torch_householder unavailable; using the equivalent native "
            "fasthpp implementation",
            stacklevel=2,
        )
        return None


def householder_orgqr(params: torch.Tensor, d: int) -> torch.Tensor:
    """Reference householder map (Bodnar et al.): torch_householder orgqr on
    compact reflectors with a unit diagonal and strictly-lower parameters.

    Args:
        params: Compact parameters of shape ``[batch_size, d*(d-1)//2]``.
        d: The dimension of the resulting orthogonal matrix.

    Returns:
        Orthogonal restriction map of shape ``[batch_size, d, d]``.
    """
    torch_householder_orgqr = _load_torch_householder_orgqr()
    if torch_householder_orgqr is None:
        return fasthpp(params, d)
    batch_size = params.size(0)
    A = torch.zeros(batch_size, d, d, device=params.device, dtype=params.dtype)
    idx = torch.tril_indices(d, d, offset=-1, device=params.device)
    A[:, idx[0], idx[1]] = params
    A = A + torch.eye(d, device=params.device, dtype=params.dtype)
    return torch_householder_orgqr(A)


def _reflection_product(
    V: torch.Tensor, d: int, stop_recursion: int | None = None
) -> torch.Tensor:
    """Product of reflections H(V[:, 0]) @ ... @ H(V[:, d-1]) via the FastH++
    parallel WY merge; V rows must be unit-normalized reflector vectors.
    """
    batch_size = V.size(0)

    D = 2 ** math.ceil(math.log2(d))

    if D > d:
        # Pad with unit-basis reflectors: they touch only the padded
        # coordinates, so the top-left d x d block is the exact product.
        V_padded = torch.zeros(batch_size, D, D, device=V.device, dtype=V.dtype)
        V_padded[:, :d, :d] = V
        for i in range(d, D):
            V_padded[:, i, i] = 1.0
        V = V_padded

    Y_ = V
    W_ = -2 * Y_.clone()

    k = 1
    num_iters = int(np.log2(D))

    for c in range(num_iters):
        k_2 = k
        k *= 2

        Y_v = Y_.view(batch_size, D // k_2, k_2, D)
        W_v = W_.view(batch_size, D // k_2, k_2, D)

        m1_ = torch.matmul(Y_v[:, 0::2], W_v[:, 1::2].transpose(-1, -2))
        m2_ = torch.matmul(W_v[:, 0::2].transpose(-1, -2), m1_)

        W_v_0 = W_v[:, 0::2]
        W_v_1 = W_v[:, 1::2] + m2_.transpose(-1, -2)

        W_ = torch.stack((W_v_0, W_v_1), dim=2).view(batch_size, D, D)

        if stop_recursion is not None and c == stop_recursion:
            break

    X = (
        torch.eye(D, dtype=V.dtype, device=V.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )

    if stop_recursion is None:
        W_out = X + torch.matmul(W_.transpose(-1, -2), torch.matmul(Y_, X))
    else:
        for i in range(D // k - 1, -1, -1):
            W_slice = W_[:, i * k : (i + 1) * k]
            Y_slice = Y_[:, i * k : (i + 1) * k]
            X = X + torch.matmul(W_slice.transpose(-1, -2), torch.matmul(Y_slice, X))
        W_out = X

    return W_out[:, :d, :d]


# Strategies accepted by build_orthogonal / the orth_strategy config option.
ORTH_STRATEGIES = ("cayley", "fasth", "fasthpp", "householder")


def orth_param_count(strategy: str, d: int) -> int:
    """Free parameters each strategy needs per restriction map."""
    if strategy == "fasth":
        return d * d
    return (d * (d - 1)) // 2


def build_orthogonal(
    params: torch.Tensor, d: int, strategy: str, clamp_val: float = 10.0
) -> torch.Tensor:
    """Dispatches raw learner outputs to the chosen orthogonal parameterization.

    cayley: half-Cayley on skew params; fasth: FastH++ on d free reflectors;
    fasthpp: FastH++ on compact reflectors; householder: reference orgqr on
    the same compact reflectors (fasthpp and householder compute the same map).
    """
    if strategy == "cayley":
        return cayley(params, d, clamp_val)
    if strategy == "fasth":
        return householder(params, d)
    if strategy == "fasthpp":
        return fasthpp(params, d)
    if strategy == "householder":
        return householder_orgqr(params, d)
    raise ValueError(
        f"orth_strategy must be one of {ORTH_STRATEGIES}, got {strategy!r}"
    )
