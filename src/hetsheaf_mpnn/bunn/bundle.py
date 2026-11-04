# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Type-aware bundle-map learners for heterogeneous Bundle Neural Networks.

In a flat bundle the only learned geometry is the per-node map
:math:`O_v = \phi(G, P, X, v)` (Eq. 1 of Bamberger et al., ICLR 2025), so the
heterogeneous context goes into :math:`\phi`, just as the HetSheaf families
put it into their restriction-map generators:

* :class:`HetSumGNNConv` -- the paper's SumGNN layer with relation-specific
  neighbour weights, so :math:`\phi` can tell which relation each neighbour
  arrives through.
* :class:`HetBundleMapLearner` -- :math:`\phi` reading any combination of
  node features, positional encodings and node-type one-hots, plus edge types
  through :class:`HetSumGNNConv`.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from sheaf_mpnn.bunn.bundle import BundleMapLearner, PhiKind, SumGNNConv


class HetSumGNNConv(SumGNNConv):
    r"""SumGNN layer with relation-specific neighbour weights.

    Extends the bundle-learner GNN of Appendix F.1 in the style of R-GCN
    (Schlichtkrull et al., 2018):

    .. math::

        h_v = W_\mathrm{s} x_v + W_\mathrm{n} \sum_{u \in N(v)} x_u
            + \sum_r W_r \sum_{u \in N_r(v)} x_u + b,

    where :math:`N_r(v)` holds the neighbours reaching ``v`` through an edge of
    type ``r``. The shared :math:`W_\mathrm{n}` keeps the type-blind SumGNN as
    the base case, and the :math:`W_r` start at zero, so the layer begins as
    that SumGNN and learns per-relation corrections. Called without
    ``edge_types`` it skips the relation term, matching the all-zero edge-type
    one-hots the other het learners fall back to.
    """

    def __init__(
        self, in_dim: int, out_dim: int, num_edge_types: int, bias: bool = True
    ):
        super().__init__(in_dim, out_dim, bias=bias)
        if num_edge_types <= 0:
            raise ValueError("num_edge_types must be positive")
        self.num_edge_types = num_edge_types
        self.rel_weight = nn.Parameter(torch.zeros(num_edge_types, in_dim, out_dim))

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.zeros_(self.rel_weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = super().forward(x, edge_index)
        if edge_types is None:
            return out
        src, dst = edge_index
        # R-GCN's trick: transform every node once per relation, then gather one
        # row per edge -- an [N, R, out] intermediate instead of [E, R, in].
        rel = torch.einsum("nc,rco->nro", x, self.rel_weight)
        return out.index_add(0, dst, rel[src, edge_types])


class HetBundleMapLearner(BundleMapLearner):
    r"""Type-aware bundle-map learner: :math:`O_v = \phi(x_v, p_v, t_v)`.

    Same maps as :class:`~sheaf_mpnn.bunn.bundle.BundleMapLearner` (direct
    O(2) angles at ``bundle_dim=2``, the shared orthogonal parameterisations
    otherwise), with :math:`\phi` reading

    * ``use_features`` -- the node context features, the homogeneous input;
    * ``pe_size`` -- positional encodings, as in Eq. 1;
    * ``use_node_types`` -- node-type one-hots (all-zero when ``node_types``
      is not passed);
    * ``use_edge_types`` -- edge types, through relation-specific neighbour
      weights in every SumGNN layer (:class:`HetSumGNNConv`). Only
      ``phi="sum_gnn"`` reads the edges at all, so the flag is rejected for the
      MLP rather than silently dropped.

    With ``use_features=False`` and an MLP, the maps depend on the node type
    alone: every node of type ``a`` gets the same frame :math:`O_a`, and the
    transport between types ``a`` and ``b`` is the learned
    :math:`O_a^\top O_b` -- a bundle set by the schema.
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
        pe_size: int = 0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
    ):
        """Initializes the type-aware bundle map learner.

        Args:
            context_dim: Width of the node context features ``x_feat``.
            num_bundles: Number of parallel bundles.
            bundle_dim: Stalk dimension ``d`` of each bundle.
            phi: ``"mlp"`` or ``"sum_gnn"``; ``use_edge_types`` needs the
                latter.
            num_layers: Number of linear (or SumGNN) layers in ``phi``.
            orth_strategy: Orthogonal parameterisation used when
                ``bundle_dim != 2``.
            clamp_val: Clamp on the raw orthogonal parameters (Cayley only).
            pe_size: Width of the positional encodings passed to ``forward``.
            num_node_types: Number of node types (used when
                ``use_node_types=True``).
            num_edge_types: Number of edge types (used when
                ``use_edge_types=True``).
            use_features: Feed the node context features to ``phi``.
            use_node_types: Feed node-type one-hots to ``phi``.
            use_edge_types: Give every SumGNN layer relation-specific
                neighbour weights.

        Raises:
            ValueError: On an inconsistent het configuration (see above), an
                empty ``phi`` input, or anything the homogeneous learner
                rejects.
        """
        if pe_size < 0:
            raise ValueError("pe_size must be non-negative")
        if use_features and context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if use_node_types and num_node_types <= 0:
            raise ValueError(
                f"num_node_types must be > 0 when use_node_types=True, "
                f"got num_node_types={num_node_types}"
            )
        if use_edge_types and num_edge_types <= 0:
            raise ValueError(
                f"num_edge_types must be > 0 when use_edge_types=True, "
                f"got num_edge_types={num_edge_types}"
            )
        if use_edge_types and phi != "sum_gnn":
            raise ValueError(
                "use_edge_types=True requires phi='sum_gnn': an MLP phi never "
                f"reads the edges, so the edge types would be dropped; got phi={phi!r}"
            )
        input_dim = (
            (context_dim if use_features else 0)
            + pe_size
            + (num_node_types if use_node_types else 0)
        )
        if input_dim <= 0:
            raise ValueError(
                "phi has no input: enable use_features or use_node_types, or "
                "pass positional encodings (pe_size > 0)"
            )

        super().__init__(
            input_dim,
            num_bundles,
            bundle_dim=bundle_dim,
            phi=phi,
            num_layers=num_layers,
            orth_strategy=orth_strategy,
            clamp_val=clamp_val,
        )
        self.pe_size = pe_size
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_features = use_features
        self.use_node_types = use_node_types
        self.use_edge_types = use_edge_types

        if use_edge_types:
            # Same widths as the SumGNN stack the base just built, with
            # relation-specific neighbour weights added to every layer.
            het_convs: list[nn.Module] = []
            for conv in self.convs:
                assert isinstance(conv, SumGNNConv)
                het_convs.append(
                    HetSumGNNConv(
                        conv.linear_self.in_features,
                        conv.linear_self.out_features,
                        num_edge_types,
                    )
                )
            self.convs = nn.ModuleList(het_convs)

    def _context(
        self,
        x_feat: torch.Tensor,
        node_types: torch.Tensor | None,
        pe: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenates ``phi``'s input: features, then PE, then node types."""
        parts: list[torch.Tensor] = []
        if self.use_features:
            parts.append(x_feat)
        if self.pe_size:
            if pe is None:
                raise ValueError(
                    f"map learner was built with pe_size={self.pe_size}; "
                    "pass pe to forward"
                )
            if pe.size(-1) != self.pe_size:
                raise ValueError(
                    f"pe has width {pe.size(-1)} but pe_size is {self.pe_size}"
                )
            parts.append(pe)
        if self.use_node_types:
            if node_types is not None:
                parts.append(
                    F.one_hot(node_types, self.num_node_types).to(x_feat.dtype)
                )
            else:
                parts.append(x_feat.new_zeros(x_feat.size(0), self.num_node_types))
        return torch.cat(parts, dim=-1)

    def _raw_params(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_edge_types:
            return super()._raw_params(x_feat, edge_index)
        h = x_feat
        for i, conv in enumerate(self.convs):
            if i > 0:
                h = F.elu(h)
            h = conv(h, edge_index, edge_types)
        return h

    def forward(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the per-node bundle maps from the het context.

        Args:
            x_feat: Node context features ``[N, context_dim]``. Always passed
                (it fixes ``N``, dtype and device) but only read when
                ``use_features=True``.
            edge_index: Graph connectivity ``[2, E]`` (SumGNN ``phi`` only).
            node_types: Node-type indices ``[N]`` or None.
            edge_types: Edge-type indices ``[E]`` or None.
            pe: Positional encodings ``[N, pe_size]``; required when
                ``pe_size > 0``.

        Returns:
            Orthogonal maps ``[N, num_bundles, bundle_dim, bundle_dim]``.
        """
        context = self._context(x_feat, node_types, pe)
        return self._maps_from_params(self._raw_params(context, edge_index, edge_types))


__all__ = ["HetBundleMapLearner", "HetSumGNNConv"]
