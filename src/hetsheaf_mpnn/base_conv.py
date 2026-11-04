# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from sheaf_mpnn.base_conv import BaseSheafConv


class HetContextMixin:
    """Heterogeneous-context machinery shared by every het convolution base.

    Hosts must set ``context_dim`` (all convolution bases in ``sheaf_mpnn``
    do) and call ``_init_het_context()`` from their ``__init__`` to store and
    validate the het configuration. The mixin then provides:

    * ``_mlp_input_dim()`` -- total MLP input width given context flags.
    * ``_build_context_pair(x_feat, edge_index, node_types, edge_types)``
      -- builds the ``[2E, mlp_input_dim]`` batched MLP input.
    * ``_ensemble_forward(x_feat, edge_index, node_types, edge_types)``
      -- routes edges through per-edge-type MLPs (TypeEnsemble strategy).
    * ``_make_map_generators(hidden_dim, out_dim)`` -- builds the standard
      two-layer generator MLP, or one per edge type in ensemble mode.
    """

    context_dim: int
    num_node_types: int
    num_edge_types: int
    use_features: bool
    use_node_types: bool
    use_edge_types: bool
    ensemble: bool

    def _init_het_context(
        self,
        num_node_types: int,
        num_edge_types: int,
        use_features: bool,
        use_node_types: bool,
        use_edge_types: bool,
        ensemble: bool,
    ) -> None:
        """Stores the het configuration and validates it eagerly so callers get
        a clear error instead of a cryptic crash inside _build_context_pair or
        map_generators.
        """
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_features = use_features
        self.use_node_types = use_node_types
        self.use_edge_types = use_edge_types
        self.ensemble = ensemble

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
        if ensemble and num_edge_types <= 0:
            raise ValueError(
                f"num_edge_types must be > 0 when ensemble=True, "
                f"got num_edge_types={num_edge_types}"
            )
        mlp_input_dim = self._mlp_input_dim()
        if mlp_input_dim <= 0:
            raise ValueError(
                "Invalid het-context configuration: computed MLP input width is 0. "
                "Enable at least one context source that contributes to the MLP "
                "input. In ensemble mode, only use_features and use_node_types "
                "contribute to the MLP input; use_edge_types only selects the "
                "per-edge-type MLP."
            )

    def _mlp_input_dim(self) -> int:
        """Width of each row in the [2E, mlp_input_dim] batched MLP input."""
        if self.ensemble:
            # Ensemble type discrimination is via MLP index.
            # Internal MLP context depends on enabled flags.
            dim = 0
            if self.use_features:
                dim += 2 * self.context_dim
            if self.use_node_types:
                dim += 2 * self.num_node_types
            return dim
        dim = 0
        if self.use_features:
            dim += 2 * self.context_dim
        if self.use_node_types:
            dim += 2 * self.num_node_types
        if self.use_edge_types:
            dim += self.num_edge_types
        return dim

    def _make_map_generators(self, hidden_dim: int, out_dim: int) -> None:
        """Builds ``map_generator`` (or ``map_generators`` in ensemble mode).

        Every generator is the standard het two-layer MLP
        ``Linear(mlp_input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim,
        out_dim)``; ensemble mode allocates one per edge type.
        """
        mlp_in = self._mlp_input_dim()

        def mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(mlp_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        if self.ensemble:
            self.map_generators = nn.ModuleList(
                [mlp() for _ in range(self.num_edge_types)]
            )
        else:
            self.map_generator = mlp()

    def _build_context_pair(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> torch.Tensor:
        """Builds a [2E, mlp_input_dim] tensor for a single batched MLP call.

        First E rows: dst-perspective  [ctx_dst, ctx_src, edge_type].
        Last  E rows: src-perspective  [ctx_src, ctx_dst, edge_type].
        """
        src_idx, dst_idx = edge_index

        parts_dst: list[torch.Tensor] = []
        parts_src: list[torch.Tensor] = []

        if self.use_features:
            parts_dst.extend([x_feat[dst_idx], x_feat[src_idx]])
            parts_src.extend([x_feat[src_idx], x_feat[dst_idx]])

        if self.use_node_types:
            if node_types is not None:
                nt_oh = F.one_hot(node_types, self.num_node_types).to(x_feat.dtype)
            else:
                nt_oh = torch.zeros(
                    x_feat.size(0),
                    self.num_node_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
            parts_dst.extend([nt_oh[dst_idx], nt_oh[src_idx]])
            parts_src.extend([nt_oh[src_idx], nt_oh[dst_idx]])

        if self.use_edge_types:
            if edge_types is not None:
                et_oh = F.one_hot(edge_types, self.num_edge_types).to(x_feat.dtype)
            else:
                et_oh = torch.zeros(
                    edge_index.size(1),
                    self.num_edge_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
            parts_dst.append(et_oh)
            parts_src.append(et_oh)

        row_dst = torch.cat(parts_dst, dim=-1)
        row_src = torch.cat(parts_src, dim=-1)
        return torch.cat([row_dst, row_src], dim=0)  # [2E, mlp_input_dim]

    def _ensemble_forward(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> torch.Tensor:
        """Routes edges through per-edge-type MLPs (TypeEnsemble strategy).

        Returns a [2E, output_per_mlp] tensor (dst-perspective first, src-perspective
        second) by grouping edges by type and dispatching each group through the
        corresponding MLP.
        """
        src_idx, dst_idx = edge_index

        parts_dst: list[torch.Tensor] = []
        parts_src: list[torch.Tensor] = []

        if self.use_features:
            parts_dst.extend([x_feat[dst_idx], x_feat[src_idx]])
            parts_src.extend([x_feat[src_idx], x_feat[dst_idx]])

        if self.use_node_types:
            if node_types is not None:
                nt_oh = F.one_hot(node_types, self.num_node_types).to(x_feat.dtype)
            else:
                nt_oh = torch.zeros(
                    x_feat.size(0),
                    self.num_node_types,
                    device=x_feat.device,
                    dtype=x_feat.dtype,
                )
            parts_dst.extend([nt_oh[dst_idx], nt_oh[src_idx]])
            parts_src.extend([nt_oh[src_idx], nt_oh[dst_idx]])

        if parts_dst:
            x_cat_dst = torch.cat(parts_dst, dim=-1)
            x_cat_src = torch.cat(parts_src, dim=-1)
            x_cat = torch.cat([x_cat_dst, x_cat_src], dim=0)  # [2E, mlp_ctx_dim]
        else:  # pragma: no cover  # unreachable: validator ensures mlp_input_dim > 0
            x_cat = torch.empty(
                2 * edge_index.size(1), 0, device=x_feat.device, dtype=x_feat.dtype
            )

        gens = self.map_generators

        if edge_types is None:
            return gens[0](x_cat)

        # Repeat edge_types for both perspectives.
        et = torch.cat([edge_types, edge_types], dim=0)  # [2E]
        unique_types, counts = torch.unique(et, return_counts=True)
        sorted_idx = torch.argsort(et)
        splits = sorted_idx.split(counts.tolist())

        first_linear = cast(nn.Linear, cast(nn.Sequential, gens[0])[-1])
        out = torch.empty(
            x_cat.size(0),
            first_linear.out_features,
            device=x_feat.device,
            dtype=x_feat.dtype,
        )
        for split, type_val in zip(splits, unique_types.tolist(), strict=False):
            out[split] = gens[type_val](x_cat[split])
        return out


class BaseHetSheafConv(HetContextMixin, BaseSheafConv):
    """Shared base for all heterogeneous sheaf convolution layers.

    Factors out parameterization and utilities identical across every variant
    in the NSD, NSP, and SheafAttn model families:

    * ``W1`` / ``W2`` -- bilateral stalk transforms (left dxd, right fxf).
    * ``sigma`` -- activation function (Tanh).
    * ``reset_parameters()`` -- Xavier init for W1, W2, and any map_generator.
    * ``_apply_stalk_transform(x)`` -- computes ``W1 @ x @ W2``.
    * ``_apply_norm(self_map, cross_map, edge_index, num_nodes)`` -- abstract;
      each concrete subclass delegates to the appropriate ``apply_*_norm``
      utility from ``sheaf_mpnn.utils``.

    Het-context utilities (NSD / NSP families):

    * ``_mlp_input_dim()``, ``_build_context_pair(...)``, and
      ``_ensemble_forward(...)`` are inherited from ``HetContextMixin``.

    Concrete subclasses must implement:

    * ``get_map_products(x_feat, edge_index, ...) -> (self_map, cross_map)``
    * ``forward(x_feat, x_stalk, edge_index, ...) -> updated stalk``
    * ``message(...)``
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: list[int],
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
    ):
        # Delegate W1, W2, sigma, context_dim, add_self_loops to BaseSheafConv using
        # the scalar max(in_channels) as the effective feature dimension.
        super().__init__(
            stalk_dim, max(in_channels), hidden_dim, context_dim, add_self_loops
        )
        # Override in_channels with the full list[int] so subclasses
        # can inspect per-type dims.
        self.in_channels = in_channels
        self._init_het_context(
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
        )

    # ------------------------------------------------------------------
    # Shared infrastructure (W1/W2, normalisation, default message)
    # ------------------------------------------------------------------

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        # Covers both the singular map_generator (NSD/NSP) and the
        # map_generators ModuleList (SheafAttn, ensemble).
        generators: list[nn.Sequential] = []
        if hasattr(self, "map_generator"):
            gen = self.map_generator
            if isinstance(gen, nn.Sequential):
                generators.append(gen)
        if hasattr(self, "map_generators"):
            map_gens = self.map_generators
            if isinstance(map_gens, nn.ModuleList):
                for gen in map_gens:
                    if isinstance(gen, nn.Sequential):
                        generators.append(gen)
        for gen in generators:
            for m in gen:
                if isinstance(m, nn.Linear):
                    # gain=0.01: warm-start near-zero so the Laplacian is off at init.
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _apply_stalk_transform(self, x):
        """Applies bilateral stalk transform: W1 @ x @ W2."""
        return torch.matmul(torch.matmul(self.W1, x), self.W2)


__all__ = ["BaseHetSheafConv", "HetContextMixin"]
