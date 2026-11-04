# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Flat vector bundle structures for Bundle Neural Networks.

A flat vector bundle (Bamberger et al., ICLR 2025) assigns to each *node* an
orthogonal map :math:`O_v \in O(d)`, and every edge transport factorises as
:math:`O_{vu} = O_v^\top O_u`.  This module provides:

* :func:`sync` / :func:`desync` -- move node signals into and out of the
  shared global frame (:math:`x \mapsto O_v x` and :math:`z \mapsto O_v^\top
  z`), the factorisation that turns the bundle heat kernel into the plain
  graph heat kernel (Lemma 3.1 of the paper).
* :class:`BundleMapLearner` -- the network :math:`\phi` of Eq. 1 computing the
  per-node maps from node context features, either as a node-level MLP or as
  the paper's SumGNN (Appendix F.1).
"""

from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from sheaf_mpnn.utils import ORTH_STRATEGIES, build_orthogonal, orth_param_count

PhiKind = Literal["mlp", "sum_gnn"]
PHI_KINDS: tuple[str, ...] = ("mlp", "sum_gnn")


def sync(x: torch.Tensor, node_rep: torch.Tensor) -> torch.Tensor:
    r"""Applies each node's bundle maps: :math:`\bar x_v = O_v x_v`.

    Args:
        x: Bundle-shaped node signals ``[N, b, d, f]`` (``b`` bundles of
            ``d``-dimensional stalks with ``f`` channels each).
        node_rep: Per-node orthogonal maps ``[N, b, d, d]``.

    Returns:
        Synchronised signals ``[N, b, d, f]`` in the global frame.
    """
    return torch.einsum("nbij,nbjf->nbif", node_rep, x)


def desync(x: torch.Tensor, node_rep: torch.Tensor) -> torch.Tensor:
    r"""Returns to local frames: :math:`x_v = O_v^\top \bar x_v`.

    The inverse of :func:`sync`; orthogonality makes it the transpose.

    Args:
        x: Synchronised signals ``[N, b, d, f]``.
        node_rep: Per-node orthogonal maps ``[N, b, d, d]``.

    Returns:
        Signals ``[N, b, d, f]`` back in each node's local frame.
    """
    return torch.einsum("nbji,nbjf->nbif", node_rep, x)


class SumGNNConv(MessagePassing):
    r"""One layer of the paper's bundle-learner GNN (Appendix F.1).

    Computes :math:`h_v = W_\mathrm{s} x_v + W_\mathrm{n} \sum_{u \in N(v)}
    x_u + b` -- a GraphSAGE-style update with *sum* aggregation, so the map
    learner sees the graph structure (the paper found this beats a pure MLP
    on the LRGB tasks).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__(aggr="add", node_dim=0)
        self.linear_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear_ngb = nn.Linear(in_dim, out_dim, bias=False)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.linear_self.reset_parameters()
        self.linear_ngb.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            x=x,
            size=(x.size(0), x.size(0)),
        )
        return self.linear_self(x) + self.linear_ngb(agg)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class BundleMapLearner(nn.Module):
    r"""Learns one orthogonal map per node and bundle (Eq. 1 of the paper).

    For 2-dimensional bundles the maps are parameterised *directly* by one
    rotation angle per bundle: because :math:`O(2)` is disconnected, half the
    bundles realise rotations :math:`r(\theta)` (det :math:`+1`) and half
    reflections :math:`r^*(\theta)` (det :math:`-1`), exactly as in Appendix E
    of the paper.  For ``bundle_dim != 2`` the maps fall back to the
    framework's shared orthogonal parameterisations (Cayley / Householder),
    which cover the paper's Householder-reflection option.
    """

    def __init__(
        self,
        context_dim: int,
        num_bundles: int,
        bundle_dim: int = 2,
        phi: PhiKind = "mlp",
        num_layers: int = 2,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
    ):
        """Initializes the per-node bundle map learner.

        Args:
            context_dim: Width of the node context features fed to ``phi``.
            num_bundles: Number of parallel 2- (or ``d``-) dimensional bundles.
            bundle_dim: Stalk dimension ``d`` of each bundle. ``2`` uses the
                direct angle parameterisation; other values use
                ``orth_strategy``.
            phi: ``"mlp"`` for a node-level MLP, ``"sum_gnn"`` for the paper's
                sum-aggregation GNN.
            num_layers: Number of linear (or SumGNN) layers in ``phi``. The
                paper's defaults are 2 for the MLP and 1-8 for the GNN.
            orth_strategy: Orthogonal parameterisation used when
                ``bundle_dim != 2``.
            clamp_val: Clamp on the raw orthogonal parameters (Cayley only).

        Raises:
            ValueError: On non-positive sizes or unknown ``phi`` /
                ``orth_strategy``.
        """
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if num_bundles <= 0:
            raise ValueError("num_bundles must be positive")
        if bundle_dim <= 0:
            raise ValueError("bundle_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if phi not in PHI_KINDS:
            raise ValueError(f"phi must be one of {PHI_KINDS}, got {phi!r}")
        if orth_strategy not in ORTH_STRATEGIES:
            raise ValueError(
                f"orth_strategy must be one of {ORTH_STRATEGIES}, got {orth_strategy!r}"
            )

        self.num_bundles = num_bundles
        self.bundle_dim = bundle_dim
        self.phi = phi
        self.orth_strategy = orth_strategy
        self.clamp_val = clamp_val

        self.direct = bundle_dim == 2
        # Direct O(2): one angle per bundle, rotations first, reflections after
        # (paper Appendix E takes num_bundles even and splits half/half).
        self.num_rotations = (num_bundles + 1) // 2
        params_per_bundle = (
            1 if self.direct else orth_param_count(orth_strategy, bundle_dim)
        )
        out_dim = num_bundles * params_per_bundle
        # Paper Appendix F: the learner's hidden width is twice its output.
        hidden = 2 * out_dim

        dims = [context_dim] + [hidden] * (num_layers - 1) + [out_dim]
        if phi == "mlp":
            layers: list[nn.Module] = []
            for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
                if i > 0:
                    layers.append(nn.ELU())
                layers.append(nn.Linear(d_in, d_out))
            self.mlp = nn.Sequential(*layers)
        else:
            self.convs = nn.ModuleList(
                SumGNNConv(d_in, d_out)
                for d_in, d_out in zip(dims[:-1], dims[1:], strict=True)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.phi == "mlp":
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
        else:
            for conv in self.convs:
                assert isinstance(conv, SumGNNConv)
                conv.reset_parameters()

    def _raw_params(self, x_feat: torch.Tensor, edge_index: torch.Tensor):
        if self.phi == "mlp":
            return self.mlp(x_feat)
        h = x_feat
        for i, conv in enumerate(self.convs):
            if i > 0:
                h = torch.nn.functional.elu(h)
            h = conv(h, edge_index)
        return h

    def _direct_o2(self, angles: torch.Tensor) -> torch.Tensor:
        """Builds O(2) maps from angles ``[N, b]``: rotations then reflections."""
        c, s = torch.cos(angles), torch.sin(angles)
        # det(+1) rotations for the first bundles, det(-1) reflections after.
        sign = torch.ones_like(c)
        sign[:, self.num_rotations :] = -1.0
        row0 = torch.stack([c, s], dim=-1)
        row1 = torch.stack([-sign * s, sign * c], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def forward(self, x_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes the per-node bundle maps.

        Args:
            x_feat: Node context features ``[N, context_dim]``.
            edge_index: Graph connectivity ``[2, E]`` (used by the SumGNN
                ``phi`` only).

        Returns:
            Orthogonal maps ``[N, num_bundles, bundle_dim, bundle_dim]``.
        """
        return self._maps_from_params(self._raw_params(x_feat, edge_index))

    def _maps_from_params(self, raw: torch.Tensor) -> torch.Tensor:
        """Turns ``phi``'s raw output ``[N, out_dim]`` into ``[N, b, d, d]`` maps."""
        if self.direct:
            return self._direct_o2(raw)
        num_nodes = raw.size(0)
        maps = build_orthogonal(
            raw.reshape(num_nodes * self.num_bundles, -1),
            self.bundle_dim,
            self.orth_strategy,
            self.clamp_val,
        )
        return maps.view(num_nodes, self.num_bundles, self.bundle_dim, self.bundle_dim)


__all__ = [
    "PHI_KINDS",
    "PhiKind",
    "SumGNNConv",
    "BundleMapLearner",
    "sync",
    "desync",
]
