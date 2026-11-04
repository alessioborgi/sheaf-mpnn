# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Edge-stalk nonlinearities :math:`\Phi` for Nonlinear Sheaf Diffusion.

NLSD (Zaghen et al., 2024) replaces the linear sheaf Laplacian
:math:`L_\mathcal{F} = \delta^\top \delta` with

.. math::

    L^\Phi_\mathcal{F} = \delta^\top \circ \Phi \circ \delta,

where :math:`\Phi` acts edge-wise on the 1-cochain space, i.e. independently on
each edge stalk :math:`\mathcal{F}(e) \cong \mathbb{R}^{d \times f}`.  Every
module here maps a batch of edge stalks ``[E, d, f]`` to a batch of the same
shape; setting :math:`\Phi = \mathrm{id}` recovers the linear NSD Laplacian.

Three families are provided, mirroring the reference NLSD implementation:

* :class:`ScalarThresholdNonlinearity` -- the hand-crafted opinion-dynamics
  :math:`\psi'` gate with one learnable threshold per layer.
* :class:`EdgeThresholdNonlinearity` -- the same gate, but with the threshold
  predicted per edge (and per channel) from the endpoint features.
* :class:`MLPNonlinearity` -- the fully learned :math:`\Phi` of the paper: an
  MLP on the flattened edge stalk with ReLU after every layer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import nn

PsiKind = Literal["linear", "cos"]
PSI_KINDS: tuple[PsiKind, ...] = ("linear", "cos")

# Slope of the sigmoid surrogate used to push gradients through the hard
# ``sq_norm < threshold`` indicator (reference value).
_STE_SLOPE = 100.0


def straight_through_indicator(
    sq_norm: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    r"""Hard ``sq_norm < threshold`` gate with a smooth surrogate gradient.

    The forward value is the exact 0/1 indicator, but the backward pass follows
    :math:`1 - \sigma(100 (\|y\|^2 - \tau))`, so the threshold :math:`\tau`
    stays trainable.  This is the reference straight-through construction
    ``hard - soft.detach() + soft``.

    Args:
        sq_norm: Squared stalk norms, broadcastable against ``threshold``.
        threshold: Positive gate threshold :math:`\tau`.

    Returns:
        Tensor with the shape of ``sq_norm * threshold``, valued in {0, 1}
        in the forward pass.
    """
    soft = 1.0 - torch.sigmoid(_STE_SLOPE * (sq_norm - threshold))
    return (sq_norm < threshold).to(soft.dtype) - soft.detach() + soft


def psi_derivative(
    edge_stalk: torch.Tensor, psi: PsiKind, threshold: torch.Tensor
) -> torch.Tensor:
    r"""Apply the opinion-dynamics gate :math:`\Phi(y) = \psi'(\|y\|^2)\, y`.

    Following Hansen & Ghrist (2021), the nonlinearity rescales each edge stalk
    by a scalar that depends only on the squared norm of that stalk, so the map
    stays odd (:math:`\Phi(-y) = -\Phi(y)`) and the resulting Laplacian keeps
    the sign structure of the linear one.

    * ``"linear"`` -- bounded-confidence dynamics: the interaction is switched
      off once the disagreement :math:`\|y\|^2` exceeds :math:`\tau`.
    * ``"cos"`` -- the derivative of a cosine potential on :math:`[0, \pi]`,
      which additionally attenuates near-agreement pairs.

    Args:
        edge_stalk: Edge-space features ``[E, d, f]``.
        psi: Which potential's derivative to apply.
        threshold: Positive threshold, broadcastable against the per-edge
            squared norms ``[E, 1, f]`` (a scalar, ``[E, 1, 1]``, or
            ``[E, 1, f]``).

    Returns:
        Rescaled edge stalks ``[E, d, f]``.

    Raises:
        ValueError: If ``psi`` is not a known potential.
    """
    if psi not in PSI_KINDS:
        raise ValueError(f"psi must be one of {PSI_KINDS}, got {psi!r}")

    # Norm over the stalk axis only: one scalar per (edge, channel).
    sq_norm = torch.linalg.vector_norm(edge_stalk, dim=1, keepdim=True) ** 2
    indicator = straight_through_indicator(sq_norm, threshold)

    if psi == "linear":
        deriv = indicator
    else:  # "cos"
        # Gate first so the cosine argument stays inside [0, pi].  The clamp
        # keeps pi/threshold finite when the threshold reaches exactly zero
        # (a trained |tau| crossing 0, or an edge-predicted threshold on two
        # identical endpoints); the gate is fully closed there anyway, and
        # 0 * inf would otherwise poison the output with NaNs.
        deriv = torch.sin(sq_norm * indicator * (math.pi / threshold.clamp_min(1e-8)))

    return edge_stalk * deriv


class BaseEdgeNonlinearity(nn.Module, ABC):
    r""":math:`\Phi: C^1(G; \mathcal{F}) \to C^1(G; \mathcal{F})`, edge-wise.

    Subclasses transform a batch of edge stalks in place.  ``x_dst``/``x_src``
    carry the endpoint node contexts so a nonlinearity may condition on the two
    nodes an edge joins; that dependence must be symmetric (see ``forward``).
    """

    @abstractmethod
    def forward(
        self,
        edge_stalk: torch.Tensor,
        x_dst: torch.Tensor,
        x_src: torch.Tensor,
    ) -> torch.Tensor:
        r"""Applies :math:`\Phi` to each edge stalk.

        The layer evaluates :math:`\phi_e` once per *directed* edge, so the two
        copies of an undirected edge arrive with ``edge_stalk`` negated and
        ``x_dst`` / ``x_src`` swapped.  Any dependence on the endpoint contexts
        must therefore be symmetric under that swap, otherwise the same edge
        would carry two unrelated :math:`\phi_e`.

        Args:
            edge_stalk: Edge features as the destination node sees them,
                ``F_dst x_dst - F_src x_src``, shape ``[E, d, f]``.
            x_dst: Destination node context ``[E, context_dim]``.
            x_src: Source node context ``[E, context_dim]``.

        Returns:
            Transformed edge features ``[E, d, f]``.
        """

    def reset_parameters(self) -> None:
        """Re-initialises any learnable state (no-op by default)."""


class IdentityNonlinearity(BaseEdgeNonlinearity):
    r""":math:`\Phi = \mathrm{id}`, which reduces NLSD exactly to linear NSD.

    Kept as the reference point of the construction (Section 4.2 of the paper:
    "the NSD model emerges as a specific case when :math:`\Phi` is the identity
    function") and used by the equivalence tests.
    """

    def forward(self, edge_stalk, x_dst, x_src):
        return edge_stalk


class ScalarThresholdNonlinearity(BaseEdgeNonlinearity):
    r"""Opinion-dynamics gate with one learnable threshold per layer.

    The threshold is stored unconstrained and used as ``|tau|``, so it can
    cross zero during training without leaving the valid domain.
    """

    def __init__(self, psi: PsiKind = "linear", init_threshold: float = 1.0):
        r"""Initializes the layer-wide threshold gate.

        Args:
            psi: Potential whose derivative gates the diffusion.
            init_threshold: Initial value of the threshold :math:`\tau`.
        """
        super().__init__()
        if psi not in PSI_KINDS:
            raise ValueError(f"psi must be one of {PSI_KINDS}, got {psi!r}")
        self.psi = psi
        self.init_threshold = init_threshold
        self.threshold = nn.Parameter(torch.tensor(float(init_threshold)))

    def reset_parameters(self) -> None:
        nn.init.constant_(self.threshold, float(self.init_threshold))

    def forward(self, edge_stalk, x_dst, x_src):
        return psi_derivative(edge_stalk, self.psi, self.threshold.abs())


class EdgeThresholdNonlinearity(BaseEdgeNonlinearity):
    r"""Opinion-dynamics gate whose threshold is predicted for every edge.

    A bias-free linear head reads the absolute endpoint difference
    ``|x_dst - x_src|`` -- symmetric in the two endpoints, so both directed
    copies of an edge agree -- and emits one threshold per edge, broadcast
    across the ``f`` feature channels.
    """

    def __init__(self, context_dim: int, psi: PsiKind = "linear"):
        """Initializes the per-edge threshold predictor.

        Args:
            context_dim: Width of the node context vectors ``x_feat``.
            psi: Potential whose derivative gates the diffusion.
        """
        super().__init__()
        if psi not in PSI_KINDS:
            raise ValueError(f"psi must be one of {PSI_KINDS}, got {psi!r}")
        self.psi = psi
        self.threshold = nn.Linear(context_dim, 1, bias=False)

    def reset_parameters(self) -> None:
        self.threshold.reset_parameters()

    def forward(self, edge_stalk, x_dst, x_src):
        # [E, 1] -> [E, 1, 1]: one threshold per edge, shared over stalk dims
        # and channels, matching the per-edge squared-norm shape [E, 1, f].
        thresholds = self.threshold((x_dst - x_src).abs()).abs().unsqueeze(-1)
        return psi_derivative(edge_stalk, self.psi, thresholds)


class MLPNonlinearity(BaseEdgeNonlinearity):
    r"""Fully learned :math:`\Phi`: an MLP on the flattened edge stalk.

    This is the data-driven nonlinearity of the paper -- rather than picking a
    potential by hand, :math:`\Phi` is a stack of bias-free linear maps on
    :math:`\mathcal{F}(e) \cong \mathbb{R}^{d f}`, each followed by a ReLU.
    Because the layers carry no bias, :math:`\Phi(0) = 0` and isolated or
    self-looping nodes stay inert.
    """

    def __init__(self, stalk_dim: int, in_channels: int, num_layers: int = 2):
        """Initializes the edge-stalk MLP.

        Args:
            stalk_dim: Full stalk dimension ``d`` handled by the layer.
            in_channels: Feature channels ``f`` inside each stalk entry.
            num_layers: Number of Linear+ReLU blocks. Must be positive.
        """
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.stalk_dim = stalk_dim
        self.in_channels = in_channels
        self.width = width = stalk_dim * in_channels
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Linear(width, width, bias=False))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, edge_stalk, x_dst, x_src):
        # Spell both shapes out: reshape(n, -1) is ambiguous on an edgeless graph.
        num_edges = edge_stalk.size(0)
        flat = edge_stalk.reshape(num_edges, self.width)
        return self.mlp(flat).view(num_edges, self.stalk_dim, self.in_channels)


__all__ = [
    "PSI_KINDS",
    "PsiKind",
    "BaseEdgeNonlinearity",
    "IdentityNonlinearity",
    "ScalarThresholdNonlinearity",
    "EdgeThresholdNonlinearity",
    "MLPNonlinearity",
    "psi_derivative",
    "straight_through_indicator",
]
