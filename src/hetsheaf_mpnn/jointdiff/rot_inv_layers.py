# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
import torch.nn.functional as F
from torch import nn

from hetsheaf_mpnn.base_conv import BaseHetSheafConv
from sheaf_mpnn.utils import apply_general_norm


class HetRotationInvariantSheafConv(BaseHetSheafConv):
    """Heterogeneous Rotation Invariant Sheaf Convolution.

    Generates restriction maps from the rotation-invariant products of the
    node features (x_dst @ x_src^T), extended with the het type context:
    node/edge-type one-hots are themselves rotation-invariant, so appending
    them to the map-generator input preserves the layer's invariance to the
    arbitrary orientation of the feature channels.

    The het flags reinterpret the standard context sources for the per-edge
    (single-perspective) generator input:

    * ``use_features`` -- include the flattened invariant product [d*d].
    * ``use_node_types`` -- append the one-hot types of both endpoints.
    * ``use_edge_types`` -- append the one-hot edge type.
    * ``ensemble`` -- route each edge through a per-edge-type generator MLP.

    With ``time_dep=True`` and ``use_features=True``, the previous layer's
    maps are folded into the invariant product before map generation; without
    features the product does not exist, so ``prev_maps`` is ignored.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_eps: bool = False,
        time_dep: bool = True,
    ):
        super().__init__(
            stalk_dim,
            [in_channels],
            hidden_dim,
            context_dim,
            False,  # add_self_loops: rot-inv layers never augment the graph.
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.add_eps = add_eps
        if add_eps:
            self.eps = nn.Parameter(torch.zeros(stalk_dim))

        self.time_dep = time_dep
        map_in = self._map_input_dim()
        out_dim = stalk_dim * stalk_dim

        def mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(map_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        if ensemble:
            self.map_generators = nn.ModuleList([mlp() for _ in range(num_edge_types)])
        else:
            self.map_generator = mlp()
        self.reset_parameters()

    def _map_input_dim(self) -> int:
        """Generator input width: one row per edge (single perspective), so the
        widths differ from the pairwise ``_mlp_input_dim`` of the other het
        families.
        """
        dim = 0
        if self.use_features:
            dim += self.stalk_dim * self.stalk_dim
        if self.use_node_types:
            dim += 2 * self.num_node_types
        if self.use_edge_types and not self.ensemble:
            # Ensemble type discrimination is via MLP index.
            dim += self.num_edge_types
        return dim

    def _build_map_input(
        self,
        feat_product: torch.Tensor | None,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Builds the [E, map_in] generator input for every edge.

        Args:
            feat_product: Flattened invariant products [E, d*d], or None when
                ``use_features`` is off.
            edge_index: Graph connectivity [2, E].
            node_types: Node-type indices [N] or None.
            edge_types: Edge-type indices [E] or None.
            ref: Reference tensor providing device/dtype.
        """
        src_idx, dst_idx = edge_index
        parts: list[torch.Tensor] = []

        if self.use_features:
            assert feat_product is not None
            parts.append(feat_product)

        if self.use_node_types:
            if node_types is not None:
                nt_oh = F.one_hot(node_types, self.num_node_types).to(ref.dtype)
            else:
                nt_oh = torch.zeros(
                    ref.size(0),
                    self.num_node_types,
                    device=ref.device,
                    dtype=ref.dtype,
                )
            parts.extend([nt_oh[dst_idx], nt_oh[src_idx]])

        if self.use_edge_types and not self.ensemble:
            if edge_types is not None:
                et_oh = F.one_hot(edge_types, self.num_edge_types).to(ref.dtype)
            else:
                et_oh = torch.zeros(
                    edge_index.size(1),
                    self.num_edge_types,
                    device=ref.device,
                    dtype=ref.dtype,
                )
            parts.append(et_oh)

        return torch.cat(parts, dim=-1)

    def _apply_generators(
        self, inp: torch.Tensor, edge_types: torch.Tensor | None
    ) -> torch.Tensor:
        """Runs the generator MLP; ensemble mode dispatches per edge type."""
        if not self.ensemble:
            return self.map_generator(inp)
        gens = self.map_generators
        if edge_types is None:
            return gens[0](inp)
        unique_types, counts = torch.unique(edge_types, return_counts=True)
        sorted_idx = torch.argsort(edge_types)
        splits = sorted_idx.split(counts.tolist())
        out = inp.new_empty(inp.size(0), self.stalk_dim * self.stalk_dim)
        for split, type_val in zip(splits, unique_types.tolist(), strict=False):
            out[split] = gens[type_val](inp[split])
        return out

    def get_map_products(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
        prev_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_edges = edge_index.size(1)
        src_idx, dst_idx = edge_index

        feat_product: torch.Tensor | None = None
        if self.use_features:
            # 1. Reshape the flat context into [N, d, f] stalks.
            x_reshaped = x_feat.view(x_feat.size(0), self.stalk_dim, -1)
            x_src = x_reshaped[src_idx]  # [E, d, f]
            x_dst = x_reshaped[dst_idx]  # [E, d, f]

            # 2. Rotation-invariant feature product: x_dst @ x_src^T [E, d, d].
            product = torch.bmm(x_dst, x_src.transpose(-2, -1))

            # Incorporate previous maps if time_dep is enabled.
            if self.time_dep and prev_maps is not None:
                product = torch.bmm(product, prev_maps)
            feat_product = product.view(num_edges, -1)

        # 3. Generate restriction maps from the invariant het context.
        inp = self._build_map_input(
            feat_product, edge_index, node_types, edge_types, x_feat
        )
        map_fwd = self._apply_generators(inp, edge_types)
        map_fwd = map_fwd.view(num_edges, self.stalk_dim, self.stalk_dim)

        # Use the transpose for the backward edge direction.
        map_bwd = map_fwd.transpose(1, 2)

        # 4. Construct sheaf Laplacian self and cross products.
        self_map = torch.bmm(map_fwd.transpose(1, 2), map_fwd)
        cross_map = torch.bmm(map_fwd.transpose(1, 2), map_bwd)

        return self_map, cross_map, map_fwd

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        return apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
        )

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        prev_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies one het rotation-invariant diffusion step.

        Returns:
            (updated stalk [N, stalk_dim, in_channels], map_fwd [E, d, d]).
        """
        num_nodes = x_stalk.size(0)
        src_idx, dst_idx = edge_index

        self_map, cross_map, map_fwd = self.get_map_products(
            x_feat, edge_index, node_types, edge_types, prev_maps
        )
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        z = self._apply_stalk_transform(x_stalk)
        z_src, z_dst = z[src_idx], z[dst_idx]
        out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_dst=z_dst,
            z_src=z_src,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )

        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip - self.alpha * self.sigma(out), map_fwd


__all__ = ["HetRotationInvariantSheafConv"]
