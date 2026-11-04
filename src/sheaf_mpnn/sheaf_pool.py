# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""SHEAFPOOL: gauge-invariant graph-level readout for sheaf neural networks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as scatter_softmax

from sheaf_mpnn.utils import batched_procrustes_align, batched_sym_matrix_pow


class SheafPool(MessagePassing):
    r"""Gauge-invariant, permutation-invariant graph-level readout (SHEAFPOOL).

    Each node carries a stalk embedding :math:`H_i \in \mathbb{R}^{d \times C}`.
    A naive flatten-and-pool is NOT gauge-invariant (Proposition A.3 in the paper).
    SheafPool fixes this with a six-stage pipeline implemented as message passing
    over a virtual super-node:

    (A) **Per-node whitening** removes the :math:`\mathrm{GL}(d)` gauge up to
        :math:`O(d)`:

        .. math::

            S_i = H_i H_i^\top + \varepsilon I, \qquad Z_i = S_i^{-1/2} H_i

    (B) **Procrustes alignment** to a learnable anchor :math:`A` removes
        :math:`O(d)` ambiguity:

        .. math::

            R_i^* = \arg\min_{R \,\in\, O(d)} \|R^\top Z_i - A\|_F^2
            \quad\Longrightarrow\quad
            Z_{i,\mathrm{aligned}} = {R_i^*}^\top Z_i

    (C) **Invariant attention** scores nodes using pre-alignment channel energies
        (invariant because :math:`u_i = \mathrm{diag}(Z_i^\top Z_i)` is
        :math:`O(d)`-invariant):

        .. math::

            u_i = \mathrm{diag}(Z_i^\top Z_i) \in \mathbb{R}^C,
            \qquad \alpha_i = \mathrm{softmax}_{\mathcal{V}}(s_\theta(u_i))

    (D) **Stalk-space pooling** via message passing to a virtual super-node:

        .. math::

            m_i = \alpha_i \cdot Z_{i,\mathrm{aligned}},
            \qquad P_g = \sum_{i \in \mathcal{V}_g} m_i

    (E) **Late receive-only token** avoids oversmoothing (token inserted after
        message-passing, not during it):

        .. math::

            \hat{T}_g = \mathrm{LayerNorm}\!\left(\phi\!\left([T,\, P_g]\right)\right)

        where :math:`\phi` is a channel-wise MLP.

    (F) **Invariant graph features**:

        .. math::

            g_g = \mathrm{diag}\!\left(\hat{T}_g^\top \hat{T}_g\right) \in \mathbb{R}^C

    The message passing view (stage D): for each graph :math:`g`, a virtual
    super-node receives edges from every real node :math:`i \in \mathcal{V}_g`.
    ``message()`` computes the weighted aligned stalk per edge; ``aggregate()``
    (add) sums them into :math:`P_g`.

    Args:
        stalk_dim: Stalk dimension :math:`d` (size of local vector space per node).
        channel_dim: Number of feature channels :math:`C`.
        hidden_dim: Hidden width of the attention and collect MLPs.
                    Defaults to ``channel_dim``.
        eps_whiten: Diagonal regulariser :math:`\varepsilon` added before whitening
                    (default 1e-4).
        eps_eigh: Eigenvalue clamp used in the inverse square root (default 1e-6).
        proper_rot: If True, restrict Procrustes to :math:`SO(d)` so
                    :math:`\det(R^*) = +1`.
    """

    def __init__(
        self,
        stalk_dim: int,
        channel_dim: int,
        hidden_dim: int | None = None,
        eps_whiten: float = 1e-4,
        eps_eigh: float = 1e-6,
        proper_rot: bool = True,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.stalk_dim = stalk_dim
        self.channel_dim = channel_dim
        self.eps_whiten = eps_whiten
        self.eps_eigh = eps_eigh
        self.proper_rot = proper_rot
        hidden = hidden_dim or channel_dim

        # Learnable anchor A in R^{d x C} for Procrustes alignment.
        self.anchor = nn.Parameter(torch.empty(stalk_dim, channel_dim))
        nn.init.normal_(self.anchor, std=0.02)

        # Attention scorer s_theta: R^C -> R.
        self.score = nn.Sequential(
            nn.Linear(channel_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Late receive-only token T in R^{1 x d x C}.
        self.token = nn.Parameter(torch.empty(1, stalk_dim, channel_dim))
        nn.init.normal_(self.token, std=0.02)

        # Channel-wise token MLP phi: R^{2C} -> R^C (operates on last dim).
        self.token_mlp = nn.Sequential(
            nn.Linear(channel_dim * 2, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, channel_dim),
        )
        self.layer_norm = nn.LayerNorm(channel_dim)

    def _prepare_node_features(
        self,
        H: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stages A-C: whitening, Procrustes alignment, attention weights.

        All computations are local (per-node) and run before ``propagate()``.

        Returns:
            Z_aligned: Whitened and aligned stalks [N, d, C].
            weights: Softmax attention weights within each graph [N].
        """
        # (A) Whitening: S_i = H_i H_i^T + eps*I,  Z_i = S_i^{-1/2} H_i.
        eye = torch.eye(self.stalk_dim, device=H.device, dtype=H.dtype).unsqueeze(0)
        S = H @ H.mT + self.eps_whiten * eye  # [N, d, d]
        S_inv_sqrt = batched_sym_matrix_pow(S, p=-0.5, eps=self.eps_eigh)  # [N, d, d]
        Z = S_inv_sqrt @ H  # [N, d, C]

        # (B) Procrustes alignment: Z_aligned = R_i*^T Z_i.
        Z_aligned = batched_procrustes_align(Z, self.anchor, proper=self.proper_rot)

        # (C) Invariant attention: u_i = diag(Z_i^T Z_i), softmax within each graph.
        u = (Z**2).sum(dim=1)  # [N, C] -- uses pre-alignment Z (O(d)-invariant)
        scores = self.score(u).squeeze(-1)  # [N]
        weights = scatter_softmax(scores, batch, dim=0)  # [N]

        return Z_aligned, weights

    def message(  # ty: ignore[invalid-method-override]
        self,
        Z_aligned_j: torch.Tensor,
        weights_j: torch.Tensor,
    ) -> torch.Tensor:
        """Stage D (per edge): weighted aligned stalk sent to the virtual super-node.

        Args:
            Z_aligned_j: Source (real node) aligned stalks [E, d, C].
            weights_j: Source (real node) attention weights [E].

        Returns:
            Weighted aligned stalks [E, d, C].
        """
        return Z_aligned_j * weights_j[:, None, None]

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool stalk embeddings to gauge-invariant graph-level features.

        Args:
            node_embeddings: Node stalk embeddings.  Either ``[N, d, C]`` (shaped)
                or ``[N, d*C]`` (flat); both forms are accepted.
            batch: Graph-assignment vector of shape ``[N]`` mapping each node to its
                graph index.  Pass ``None`` for a single-graph forward pass.

        Returns:
            Graph-level feature vectors of shape ``[B, C]``.
        """
        N = node_embeddings.size(0)
        if batch is None:
            batch = node_embeddings.new_zeros(N, dtype=torch.long)
        B = int(batch.max().item()) + 1

        # Accept [N, d*C] or [N, d, C].
        H = (
            node_embeddings.reshape(N, self.stalk_dim, self.channel_dim)
            if node_embeddings.dim() == 2
            else node_embeddings
        )

        # Stages A-C: local per-node preprocessing (before message passing).
        Z_aligned, weights = self._prepare_node_features(H, batch)

        # Build bipartite edges: real node i -> virtual super-node (N + batch[i]).
        # Virtual super-nodes are indexed N, N+1, ..., N+B-1.
        src = torch.arange(N, device=H.device)
        dst = batch + N
        edge_index = torch.stack([src, dst], dim=0)  # [2, N]

        # Stage D: message passing -- scatter weighted aligned stalks to super-nodes.
        # Equivalent to scatter_add(Z_aligned * weights[:,None,None], batch, dim=0).
        # size=(N, N+B): source nodes are 0..N-1, target nodes are 0..N+B-1.
        # out[0..N-1] = 0 (real nodes receive nothing), out[N..N+B-1] = P_g.
        out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            size=(N, N + B),
            Z_aligned=Z_aligned,  # [N, d, C] -- only _j (source) is needed
            weights=weights,  # [N]      -- only _j (source) is needed
        )  # [N+B, d, C]
        P = out[N:]  # [B, d, C]

        # Stage E: late token update -- T_hat_g = LayerNorm(phi([T, P_g])).
        token = self.token.expand(B, -1, -1)  # [B, d, C]
        combined = torch.cat([token, P], dim=-1)  # [B, d, 2C]
        updated = self.layer_norm(self.token_mlp(combined))  # [B, d, C]

        # Stage F: invariant graph features -- g_g = diag(T_hat_g^T T_hat_g).
        # Equivalent to diag(updated.mT @ updated) but avoids the full
        # [B, d, d] product; element-wise **2 + sum is O(d*C) vs O(d^2*C).
        return (updated**2).sum(dim=1)  # [B, C]
