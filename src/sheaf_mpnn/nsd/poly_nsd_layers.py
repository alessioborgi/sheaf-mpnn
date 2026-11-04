# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from sheaf_mpnn.nsd.nsd_layers import (
    DiagonalNSDConv,
    GeneralNSDConv,
    LowRankNSDConv,
    OrthogonalNSDConv,
)

if TYPE_CHECKING:
    from typing import Protocol

    class _NSDConvProto(Protocol):
        stalk_dim: int
        sigma: Any
        add_self_loops: bool

        def reset_parameters(self) -> None: ...
        def _apply_stalk_transform(self, x: torch.Tensor) -> torch.Tensor: ...
        def get_map_products(
            self, x_feat: torch.Tensor, edge_index: torch.Tensor
        ) -> tuple[torch.Tensor | None, torch.Tensor]: ...
        def _apply_norm(
            self,
            self_map: torch.Tensor | None,
            cross_map: torch.Tensor,
            edge_index: torch.Tensor,
            num_nodes: int,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...
        def propagate(
            self, edge_index: torch.Tensor, **kwargs: Any
        ) -> torch.Tensor: ...

    _PolyNSDBase = _NSDConvProto
else:
    _PolyNSDBase = object


class BasePolyNSDConv(_PolyNSDBase):
    """Mixin: replaces the first-order NSD forward with a degree-K polynomial filter.

    Usage: place before the NSD conv in the MRO, e.g.::

        class DiagonalPolyNSDConv(BasePolyNSDConv, DiagonalNSDConv): ...

    Captures degree/lambda_max/hp_alpha_init and forwards the rest to the NSD
    parent via cooperative super().__init__.  Output::

        poly  = sum_k softmax(theta_raw)_k * T_k(L_tilde)   (Chebyshev recurrence)
        x_out = (1 + tanh(eps)) * x_stalk
              - sigma(poly + hp_alpha * (z - Delta_F(z)/lambda_max))

    where L_tilde = (2/lambda_max)*Delta_F - I is the spectrally rescaled sheaf
    Laplacian (spectrum [-1, 1]), required for Chebyshev stability.  Delta_F(z)
    is the raw
    normalised sheaf Laplacian (spectrum [0, lambda_max]) reused for the HP skip.

    Total propagation cost: degree calls (one for the initial Delta_F(z), plus
    degree - 1 more for T2...T_{degree} via L_tilde).
    """

    def __init__(
        self,
        *args,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        **kwargs,
    ):
        """Cooperative init: intercepts polynomial hyper-params, forwards the rest.

        Args:
            *args: Positional arguments forwarded verbatim to the next MRO class.
            degree: Polynomial degree K (must be >= 1).  Determines the number of
                propagate calls per forward (degree total: one for Delta_F(z),
                shared with the high-pass skip, plus degree - 1 for the
                remaining Chebyshev recurrence steps).
            lambda_max: Spectral upper bound of the normalised sheaf Laplacian.
                For the symmetric normalisation used here lambda_max = 2.0 is
                tight; only change this when using an unnormalised Laplacian.
            hp_alpha_init: Initial value of the learnable high-pass skip weight.
                Default 0.0 means the HP skip has no effect at initialisation,
                letting the polynomial filter dominate early training.
            **kwargs: Keyword arguments forwarded to the next MRO class.
        """
        # Forward all non-polynomial args down the cooperative MRO chain.
        # This triggers DiagonalNSDConv -> BaseNSDConv -> BaseSheafConv ->
        # MessagePassing.__init__, setting self.stalk_dim, W1, W2, sigma,
        # alpha, and map_generator before we append our own parameters.
        super().__init__(*args, **kwargs)

        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = degree
        self.lambda_max = lambda_max

        # Raw coefficients for the degree-K polynomial basis.  Softmax at
        # forward time ensures the spectral response stays bounded (convex
        # combination of bounded Chebyshev basis values).
        self.theta_raw = nn.Parameter(torch.zeros(degree + 1))

        # Learnable scalar weight for the high-pass skip correction.
        # Zero-init: no HP contribution at the start of training.
        self.hp_alpha = nn.Parameter(torch.tensor(float(hp_alpha_init)))

        # Per-stalk-dimension gate bias; gate = 1 + tanh(eps).
        # eps = 0 => gate = 1, so the initial update is x - sigma(result).
        self.eps = nn.Parameter(torch.zeros(self.stalk_dim))

    # ------------------------------------------------------------------
    # Parameter reset
    # ------------------------------------------------------------------

    def reset_parameters(self):
        """Resets all parameters, including the polynomial-specific ones."""
        # Call the NSD parent's reset (Xavier init for W1, W2, map_generator).
        super().reset_parameters()
        # Polynomial params are intentionally zero-initialised; re-apply on reset.
        # Guards with hasattr because reset_parameters is also called from inside
        # DiagonalNSDConv.__init__ (via self.reset_parameters) before the mixin
        # has had a chance to set these attributes.
        if hasattr(self, "theta_raw"):
            nn.init.zeros_(self.theta_raw)
        if hasattr(self, "hp_alpha"):
            nn.init.zeros_(self.hp_alpha)
        if hasattr(self, "eps"):
            nn.init.zeros_(self.eps)

    # ------------------------------------------------------------------
    # Polynomial forward (replaces BaseNSDConv.forward)
    # ------------------------------------------------------------------

    def forward(self, x_feat, x_stalk, edge_index):
        """Applies one PolyNSD diffusion step.

        Args:
            x_feat: Node context features [N, context_dim].
            x_stalk: Lifted node features [N, d, f].
            edge_index: Graph connectivity in COO format [2, E].

        Returns:
            torch.Tensor: Updated stalk features [N, d, f].
        """
        # Apply bilateral stalk transform W1 @ x_stalk @ W2.
        z = self._apply_stalk_transform(x_stalk)  # [N, d, f]
        num_nodes = x_stalk.size(0)

        # Mirror BaseNSDConv.forward: augment with self-loops before degree
        # normalisation so isolated nodes get a non-zero degree (avoids
        # divide-by-zero / NaNs and matches first-order NSD normalisation).
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        src_idx, dst_idx = edge_index

        # Compute restriction-map products once; they are fixed for all
        # recurrence steps within this forward call.
        # For orthogonal maps, get_map_products returns (None, cross_map);
        # _apply_norm handles the None self_map internally via apply_orthogonal_norm.
        self_map, cross_map = self.get_map_products(x_feat, edge_index)
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        # Softmax-normalised polynomial weights theta_0, ..., theta_K.
        # The Chebyshev basis T_k is evaluated on L_tilde, whose spectrum is
        # rescaled to [-1, 1] (where |T_k| <= 1).  Softmax therefore guarantees
        # |p_theta(lambda)| <= 1 (convex combination of bounded basis values).
        theta = torch.softmax(self.theta_raw, dim=0)  # [K+1]

        # Raw normalised sheaf Laplacian Delta_F: spectrum in [0, lambda_max].
        # norm_self and norm_cross are captured from the enclosing scope;
        # only the input feature tensor h varies across recurrence steps.
        def _L(h: torch.Tensor) -> torch.Tensor:
            return self.propagate(
                edge_index,
                z_dst=h[dst_idx],
                z_src=h[src_idx],
                self_map=norm_self,
                cross_map=norm_cross,
                size=(num_nodes, num_nodes),
            )

        # Rescaled Laplacian L_tilde = (2/lambda_max)*Delta_F - I, spectrum [-1, 1].
        # Chebyshev polynomials are only bounded/stable on [-1, 1].
        def _L_tilde(h: torch.Tensor) -> torch.Tensor:
            return (2.0 / self.lambda_max) * _L(h) - h

        # Delta_F(z) once: shared between HP skip (raw Delta_F) and L_tilde(z) for T1.
        Lz = _L(z)  # [N, d, f]
        L_tilde_z = (2.0 / self.lambda_max) * Lz - z  # L_tilde(z), spectrum [-1, 1]

        # --- Three-term Chebyshev recurrence ---
        # T0 = z, T1 = L_tildez, T_{k+1} = 2 L_tilde T_k - T_{k-1}
        T_prev, T_curr = z, L_tilde_z
        result = theta[0] * T_prev + theta[1] * T_curr

        for k in range(2, self.degree + 1):
            T_next = 2.0 * _L_tilde(T_curr) - T_prev
            result = result + theta[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        # --- High-pass (HP) skip correction ---
        # h_hp = z - Delta_F(z)/lambda_max = (I - Delta_F/lambda_max)z  (raw Laplacian).
        result = result + self.hp_alpha * (z - Lz / self.lambda_max)

        # --- Gated residual update ---
        # gate = 1 + tanh(eps),  shape [1, d, 1], broadcasts over [N, d, f].
        # At eps = 0 (init), gate = 1 and the update is x_stalk - sigma(result).
        gate = (1.0 + torch.tanh(self.eps)).view(1, -1, 1)
        return gate * x_stalk - self.sigma(result)


# ---------------------------------------------------------------------------
# Concrete polynomial NSD variants
#
# Each class is a one-liner: it inherits the polynomial forward from
# BasePolyNSDConv and the restriction-map logic (map_generator, get_map_products,
# _apply_norm, message) from the matching first-order NSD conv class.  The
# __init__ only needs to pass arguments to the cooperative super() chain.
# ---------------------------------------------------------------------------


class DiagonalPolyNSDConv(BasePolyNSDConv, DiagonalNSDConv):
    """Polynomial NSD with diagonal restriction maps.

    Restriction maps are scalar-valued per stalk dimension (d free parameters
    per half-edge, rather than d^2 for the general case).  The self-map products
    are d-vectors and the message function applies element-wise multiplication
    instead of a full matrix multiply.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        # BasePolyNSDConv captures polynomial params, forwards the rest to
        # DiagonalNSDConv -> BaseNSDConv -> BaseSheafConv.
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=sparse_learner,
        )


class GeneralPolyNSDConv(BasePolyNSDConv, GeneralNSDConv):
    """Polynomial NSD with general (full d x d) restriction maps.

    Each restriction map is an unconstrained d x d matrix, giving the most
    expressive option at the cost of d^2 parameters per half-edge.  Supports
    an optional attention-based map initialisation.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        use_attention: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            use_attention=use_attention,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=sparse_learner,
        )


class OrthogonalPolyNSDConv(BasePolyNSDConv, OrthogonalNSDConv):
    """Polynomial NSD with orthogonal restriction maps.

    Restriction maps lie in O(d) parameterised by the Cayley transform
    (skew-symmetric entries) or the Householder decomposition (FastH).
    Because F^T F = I for orthogonal maps, the degree matrix reduces to
    a scalar multiple of the identity, so the self-map per edge is a
    scalar [E, 1, 1] rather than a full [E, d, d] matrix.

    Note: get_map_products returns (None, cross_map) for this variant;
    BasePolyNSDConv.forward handles the None self_map transparently because
    _apply_norm for orthogonal maps calls apply_orthogonal_norm which ignores
    the self_map argument entirely.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        clamp_val: float = 10.0,
        use_attention: bool = False,
        orth_strategy: str = "cayley",
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        use_edge_weights: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            clamp_val=clamp_val,
            use_attention=use_attention,
            orth_strategy=orth_strategy,
            use_edge_weights=use_edge_weights,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=sparse_learner,
        )


class LowRankPolyNSDConv(BasePolyNSDConv, LowRankNSDConv):
    """Polynomial NSD with low-rank restriction maps.

    Each map F = A @ B^T where A, B in R^{d x r}.  This bounds the effective
    rank to at most ``rank`` and uses 2*d*r parameters per half-edge instead of
    d^2, sitting between the diagonal and general variants in terms of both
    expressiveness and cost.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        degree: int = 3,
        lambda_max: float = 2.0,
        hp_alpha_init: float = 0.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        rank: int = 1,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
            rank=rank,
            degree=degree,
            lambda_max=lambda_max,
            hp_alpha_init=hp_alpha_init,
            add_lp=add_lp,
            add_hp=add_hp,
            sparse_learner=sparse_learner,
        )


__all__ = [
    "BasePolyNSDConv",
    "DiagonalPolyNSDConv",
    "GeneralPolyNSDConv",
    "OrthogonalPolyNSDConv",
    "LowRankPolyNSDConv",
]
