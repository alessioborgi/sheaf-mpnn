# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Variational sheaf learners and Cayley distributions on :math:`SO(n)`.

Implements the probabilistic machinery of Bayesian Sheaf Neural Networks
(Gillespie et al., 2025, arXiv:2410.09590): the restriction maps of the
cellular sheaf are latent random variables whose variational distribution is
parameterized by a learner conditioned on the node features.

Four restriction-map families are supported, mirroring the deterministic NSD
variants:

* **diagonal** -- entries are independent Gaussians
  :math:`\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))` with a standard
  normal prior (paper Case 2).
* **general** -- all :math:`d^2` entries are independent Gaussians with a
  standard normal prior (paper Case 1).
* **low_rank** -- a framework extension of Case 1 to the factored maps of
  ``LowRankNSDConv``: the :math:`2dr` entries of the factors :math:`A, B \in
  \mathbb{R}^{d \times r}` of :math:`F = AB^\top` are independent Gaussians
  with a standard normal prior. The noise lives in the factors, so every
  sampled map is rank :math:`\le r` by construction (sampling in map space
  would destroy the rank bound).
* **orthogonal** -- maps live in :math:`SO(d)` and follow the *Cayley
  distribution* :math:`\mathcal{C}_d(M, \kappa)` (paper Definition 8) with a
  uniform prior on :math:`SO(d)`.

The Cayley distribution is reparameterizable: for :math:`X` uniform on
:math:`SO(d)`,

.. math::

    Y = C\Big(\tfrac{1-\kappa}{1+\kappa}\, C^{-1}(X)\Big)\, M
    \sim \mathcal{C}_d(M, \kappa),

where :math:`C` is the (involutive form of the) Cayley transform. Closed-form
KL divergences from the uniform prior exist for :math:`d = 2, 3` (paper
Proposition 3); larger stalk dimensions fall back to a single-sample Monte
Carlo estimate through the closed-form density (paper Theorem 1), matching the
reference implementation.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from sheaf_mpnn.utils import build_orthogonal, orth_param_count

# Bound the SO(d) concentration scale away from 0 and 1: at 0 the sampled
# rotation collapses onto the mean (KL explodes), at 1 the distribution is
# uniform and gradients vanish (reference bound: sigmoid * 0.95 + 0.025).
_SCALE_MARGIN = 0.05


# ---------------------------------------------------------------------------
# Cayley transform and uniform SO(n) sampling
# ---------------------------------------------------------------------------


def cayley_involution(mats: torch.Tensor) -> torch.Tensor:
    r"""Involutive Cayley transform :math:`A \mapsto (I + A)^{-1}(I - A)`.

    Maps skew-symmetric matrices to :math:`SO(n)` and, being self-inverse,
    maps special orthogonal matrices (without eigenvalue :math:`-1`) back to
    skew-symmetric matrices. Batched over leading dimensions.

    Args:
        mats: Batch of square matrices ``[..., n, n]`` (skew-symmetric or
            special orthogonal).

    Returns:
        The transformed batch ``[..., n, n]``.
    """
    eye = torch.eye(mats.size(-1), device=mats.device, dtype=mats.dtype)
    return torch.linalg.solve(eye + mats, eye - mats)


def sample_uniform_so(
    num_samples: int,
    d: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Draw Haar-uniform samples from :math:`SO(d)`.

    QR decomposition of a Gaussian matrix with the sign correction of
    Mezzadri (2007) yields uniform :math:`O(d)`; negating the last column of
    determinant :math:`-1` samples folds them onto :math:`SO(d)`.

    Args:
        num_samples: Number of rotations to draw.
        d: Rotation dimension.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Uniform rotations ``[num_samples, d, d]`` (no gradient).
    """
    gauss = torch.randn(num_samples, d, d, device=device, dtype=dtype)
    q, r = torch.linalg.qr(gauss)
    # Fix the sign ambiguity of QR so the distribution is exactly Haar.
    sign = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    flip = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    q[..., -1] = q[..., -1] * flip.unsqueeze(-1)
    return q


def sample_cayley_rotations(
    mean_maps: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    r"""Reparameterized sample from the Cayley distribution (Definition 8).

    Draws :math:`X` uniform on :math:`SO(d)`, contracts its skew-symmetric
    Cayley preimage by ``scale`` :math:`= (1-\kappa)/(1+\kappa)`, maps back to
    :math:`SO(d)`, and composes with the mean rotation. Gradients flow into
    both ``mean_maps`` and ``scale``; the uniform noise carries none.

    Args:
        mean_maps: Mean rotations :math:`M` ``[N, d, d]``.
        scale: Concentration scales in :math:`(0, 1]` ``[N, 1]`` (1 recovers
            the uniform distribution, values near 0 concentrate on the mean).

    Returns:
        Sampled rotations ``[N, d, d]``.
    """
    n, d = mean_maps.size(0), mean_maps.size(-1)
    noise = sample_uniform_so(n, d, device=mean_maps.device, dtype=mean_maps.dtype)
    skew = cayley_involution(noise)
    perturbation = cayley_involution(scale.view(-1, 1, 1) * skew)
    return torch.matmul(perturbation, mean_maps)


# ---------------------------------------------------------------------------
# KL divergences
# ---------------------------------------------------------------------------


def gaussian_kl(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    r"""KL of :math:`\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))` from
    :math:`\mathcal{N}(0, I)`, summed over map entries, averaged over maps.

    Args:
        mean: Gaussian means ``[N, k]``.
        log_var: Gaussian log-variances ``[N, k]``.

    Returns:
        Scalar KL divergence estimate.
    """
    inner = -0.5 * (1.0 + log_var - mean.pow(2) - log_var.exp())
    return inner.sum(dim=-1).mean()


def cayley_log_density(
    rotations: torch.Tensor, kappa: torch.Tensor, d: int
) -> torch.Tensor:
    r"""Log density of :math:`\mathcal{C}_d(I, \kappa)` w.r.t. the normalized
    Haar measure (paper Theorem 1):

    .. math::

        \log f_d(P) = \tfrac{d(d-1)}{2} \log(1 - \kappa^2)
                      + (1 - d) \log\det(P - \kappa I).

    ``det(P - kappa I)`` is strictly positive for :math:`P \in SO(d)` and
    :math:`0 \le \kappa < 1` (unit-circle eigenvalues stay at distance
    :math:`\ge 1 - \kappa` from :math:`\kappa`), so ``slogdet`` is safe.

    Args:
        rotations: Rotations ``[N, d, d]`` (density is evaluated at
            :math:`P M^\top`, so pre-multiply by :math:`M^\top` for a
            non-identity mean).
        kappa: Concentration parameters in :math:`[0, 1)` ``[N]`` or ``[N, 1]``.
        d: Rotation dimension.

    Returns:
        Per-sample log densities ``[N]``.
    """
    kappa = kappa.reshape(-1)
    eye = torch.eye(d, device=rotations.device, dtype=rotations.dtype)
    _, logdet = torch.linalg.slogdet(rotations - kappa.view(-1, 1, 1) * eye)
    return 0.5 * d * (d - 1) * torch.log1p(-kappa.pow(2)) + (1 - d) * logdet


def cayley_kl(scale: torch.Tensor, d: int) -> torch.Tensor:
    r"""KL of :math:`\mathcal{C}_d(M, \kappa)` from the uniform distribution
    on :math:`SO(d)`, averaged over maps.

    Uses the closed forms of paper Proposition 3 for :math:`d = 2, 3`:

    .. math::

        \mathrm{KL}_2 &= -\log(1 - \kappa^2), \\
        \mathrm{KL}_3 &= -\log(1 - \kappa^2) - 2\log(1 - \kappa) - 2\kappa,

    and a single-sample Monte Carlo estimate through the closed-form density
    for :math:`d \ge 4` (the mean over thousands of edge maps keeps the
    estimator's variance low). The KL does not depend on the mean rotation.

    Args:
        scale: Concentration scales :math:`(1-\kappa)/(1+\kappa)` in
            :math:`(0, 1]` ``[N, 1]``.
        d: Rotation dimension.

    Returns:
        Scalar KL divergence (estimate for :math:`d \ge 4`).
    """
    kappa = (1.0 - scale) / (1.0 + scale)
    if d == 2:
        return (-torch.log1p(-kappa.pow(2))).mean()
    if d == 3:
        return (
            -torch.log1p(-kappa.pow(2)) - 2.0 * torch.log1p(-kappa) - 2.0 * kappa
        ).mean()
    eye = torch.eye(d, device=scale.device, dtype=scale.dtype)
    samples = sample_cayley_rotations(eye.expand(scale.size(0), d, d), scale)
    return cayley_log_density(samples, kappa, d).mean()


# ---------------------------------------------------------------------------
# Variational sheaf learners
# ---------------------------------------------------------------------------


class BaseVariationalSheafLearner(nn.Module):
    r"""Base class for variational sheaf learners.

    Maps the concatenated features of each incident node-edge pair to
    variational distribution parameters (paper Eq. 21), from which i.i.d.
    restriction-map samples are drawn -- one sheaf per diffusion layer. The
    parameters are computed **once** per forward pass; only the sampling is
    repeated, so all layer samples share the same posterior.

    Following the reference learner, the mean and concentration heads are
    bias-free linear layers on ``[x_u || x_u']``; ``sparse`` switches to the
    stalk-summed input reduction of ``LocalConcatSheafLearnerVariant``
    (requires ``context_dim == stalk_dim * in_channels``).

    Subclasses define ``mean_size`` / ``scale_size`` (head widths per map),
    ``sample(params)``, and ``kl(params)``.
    """

    mean_size: int
    scale_size: int

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        sparse: bool = False,
    ):
        super().__init__()
        self.d = stalk_dim
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.sparse = sparse
        in_dim = 2 * in_channels if sparse else 2 * context_dim
        self.lin_mean = nn.Linear(in_dim, self.mean_size, bias=False)
        self.lin_scale = nn.Linear(in_dim, self.scale_size, bias=False)

    def reset_parameters(self) -> None:
        self.lin_mean.reset_parameters()
        self.lin_scale.reset_parameters()

    def _pair_input(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Concatenated ``[x_src || x_dst]`` per directed edge; row ``i``
        parameterizes the map of the incident pair ``src_i`` ⊴ ``e_i``.
        """
        src_idx, dst_idx = edge_index
        inp = torch.cat([x_feat[src_idx], x_feat[dst_idx]], dim=-1)
        if self.sparse:
            inp = inp.view(inp.size(0), -1, 2 * self.in_channels).sum(dim=1)
        return inp

    def forward(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute variational parameters for every incident node-edge pair.

        Args:
            x_feat: Node context features ``[N, context_dim]``.
            edge_index: Graph connectivity ``[2, E]`` (both orientations of
                every undirected edge).

        Returns:
            ``(mean_params, scale_params)`` -- raw head outputs, transformed
            by the subclass in ``sample`` / ``kl``.
        """
        inp = self._pair_input(x_feat, edge_index)
        return self.lin_mean(inp), self.lin_scale(inp)

    def sample(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Draw one restriction map per incident node-edge pair."""
        raise NotImplementedError

    def kl(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """KL divergence of the variational posterior from the prior."""
        raise NotImplementedError


class DiagonalVariationalSheafLearner(BaseVariationalSheafLearner):
    r"""Gaussian posterior over invertible-diagonal restriction maps.

    Each of the ``d`` diagonal entries follows
    :math:`\mathcal{N}(\tanh(\mu), \sigma^2)` with a standard normal prior
    (paper Case 2). The log-variance head is bounded above via
    :math:`-\mathrm{ELU}(-x)` as in the reference, keeping :math:`\log
    \sigma^2 < 1` while leaving it unbounded below.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        sparse: bool = False,
    ):
        self.mean_size = stalk_dim
        self.scale_size = stalk_dim
        super().__init__(stalk_dim, in_channels, context_dim, sparse)

    def _transform(
        self, params: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mean, raw_scale = params
        return torch.tanh(raw_mean), -F.elu(-raw_scale)

    def sample(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, log_var = self._transform(params)
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn_like(std)  # [E, d]

    def kl(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, log_var = self._transform(params)
        return gaussian_kl(mean, log_var)


class GeneralVariationalSheafLearner(DiagonalVariationalSheafLearner):
    r"""Gaussian posterior over general linear restriction maps.

    All :math:`d^2` entries follow independent Gaussians with a standard
    normal prior (paper Case 1); samples are reshaped to ``[E, d, d]``.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        sparse: bool = False,
    ):
        # Bypass the diagonal constructor's head sizing: a general map has
        # d*d free entries.
        self.mean_size = stalk_dim * stalk_dim
        self.scale_size = stalk_dim * stalk_dim
        BaseVariationalSheafLearner.__init__(
            self, stalk_dim, in_channels, context_dim, sparse
        )

    def sample(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        flat = super().sample(params)
        return flat.view(-1, self.d, self.d)  # [E, d, d]


class LowRankVariationalSheafLearner(DiagonalVariationalSheafLearner):
    r"""Gaussian posterior over low-rank restriction maps :math:`F = AB^\top`.

    The :math:`2dr` entries of the factors :math:`A, B \in \mathbb{R}^{d
    \times r}` follow independent Gaussians with a standard normal prior --
    the Case 1 treatment applied to the factors of ``LowRankNSDConv`` rather
    than to the map entries. Sampling the factors (not the composed map)
    guarantees every drawn restriction map has rank at most ``rank``; samples
    keep factor shape ``[E, d, 2r]`` (left factor ``A`` then right factor
    ``B`` along the last axis) so the convolution can form the Laplacian
    products without materializing :math:`F`.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        sparse: bool = False,
        rank: int = 1,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        # Bypass the diagonal constructor's head sizing: a rank-r map has
        # 2*d*r free factor entries.
        self.mean_size = 2 * stalk_dim * rank
        self.scale_size = 2 * stalk_dim * rank
        BaseVariationalSheafLearner.__init__(
            self, stalk_dim, in_channels, context_dim, sparse
        )
        self.rank = rank

    def sample(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        flat = super().sample(params)
        return flat.view(-1, self.d, 2 * self.rank)  # [E, d, 2r]


class OrthogonalVariationalSheafLearner(BaseVariationalSheafLearner):
    r"""Cayley-distribution posterior over :math:`SO(d)` restriction maps.

    The mean head outputs parameters of the chosen orthogonal transform
    (tanh-bounded, as in the deterministic NSD learner); the concentration
    head outputs a single scalar per map, squashed to the scale
    :math:`s = (1-\kappa)/(1+\kappa) \in (0.025, 0.975)`. Samples follow
    :math:`\mathcal{C}_d(M, \kappa)` (paper Case 3) and the prior is uniform
    on :math:`SO(d)`.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        context_dim: int,
        sparse: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
    ):
        self.mean_size = orth_param_count(orth_strategy, stalk_dim)
        self.scale_size = 1
        super().__init__(stalk_dim, in_channels, context_dim, sparse)
        self.orth_strategy = orth_strategy
        self.clamp_val = clamp_val

    def _transform(
        self, params: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_mean, raw_scale = params
        mean_maps = build_orthogonal(
            torch.tanh(raw_mean), self.d, self.orth_strategy, self.clamp_val
        )
        scale = torch.sigmoid(raw_scale) * (1.0 - _SCALE_MARGIN) + _SCALE_MARGIN / 2
        return mean_maps, scale

    def sample(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean_maps, scale = self._transform(params)
        return sample_cayley_rotations(mean_maps, scale)  # [E, d, d]

    def kl(self, params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, scale = self._transform(params)
        return cayley_kl(scale, self.d)


__all__ = [
    "BaseVariationalSheafLearner",
    "DiagonalVariationalSheafLearner",
    "GeneralVariationalSheafLearner",
    "LowRankVariationalSheafLearner",
    "OrthogonalVariationalSheafLearner",
    "cayley_involution",
    "cayley_kl",
    "cayley_log_density",
    "gaussian_kl",
    "sample_cayley_rotations",
    "sample_uniform_so",
]
