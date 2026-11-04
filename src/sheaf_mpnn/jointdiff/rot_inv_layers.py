import torch
from torch import nn

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.utils import apply_general_norm


class RotationInvariantSheafConv(BaseSheafConv):
    """Rotation Invariant Sheaf Convolution.

    Generates restriction maps strictly from the rotation-invariant
    products of the node features (x_dst @ x_src^T), making it independent
    of the arbitrary orientation of the feature channels.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
        time_dep: bool = True,
    ):
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim,
            add_self_loops,
            add_lp,
            add_hp,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.add_eps = add_eps
        self.dropout_layer = nn.Dropout(dropout)

        if self.add_eps:
            self.eps = nn.Parameter(torch.zeros(self.stalk_dim))

        self.time_dep = time_dep
        # The map generator directly takes the flattened [d, d] feature product
        self.map_generator = nn.Sequential(
            nn.Linear(self.d * self.d, self.d * self.d, bias=True),
            self.sigma,
        )
        self.reset_parameters()

    def get_map_products(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        prev_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = edge_index.size(1)
        src_idx, dst_idx = edge_index

        # 1. Reshape raw features into [num_nodes, final_d, f] stalks and
        # select the learned dims
        x_reshaped = x_feat.view(-1, self.stalk_dim, self.in_channels)[:, : self.d, :]

        x_src = x_reshaped[src_idx]  # [E, d, f]
        x_dst = x_reshaped[dst_idx]  # [E, d, f]

        # 2. Compute the rotation-invariant feature product: x_dst @ x_src^T
        # This reduces the f dimension, yielding a structural [E, d, d] matrix
        feat_product = torch.bmm(x_dst, x_src.transpose(-2, -1))

        # Incorporate previous maps if time_dep is enabled
        if self.time_dep and prev_maps is not None:
            # We multiply feat_product by prev_maps to evolve the maps
            feat_product = torch.bmm(feat_product, prev_maps)

        # 3. Generate restriction maps from the invariant products
        map_fwd = self.map_generator(feat_product.view(E, -1))
        map_fwd = map_fwd.view(E, self.d, self.d)

        # Use the transpose for the backward edge direction
        map_bwd = map_fwd.transpose(1, 2)

        # 4. Construct Sheaf Laplacian cross and self products
        self_map = torch.bmm(map_fwd.transpose(1, 2), map_fwd)
        cross_map = torch.bmm(map_fwd.transpose(1, 2), map_bwd)

        return self_map, cross_map, map_fwd

    def _apply_norm(
        self,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ):
        return apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
            learned_dim=self.d,
        )

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        prev_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x_feat.size(0)
        src_idx, dst_idx = edge_index

        # Evaluate map products natively per-edge
        self_map, cross_map, map_fwd = self.get_map_products(
            x_feat, edge_index, prev_maps
        )
        self_map, cross_map = self._append_fixed_block(self_map, cross_map)

        # Normalize and propagate
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        # Dropout hits only the diffusion input; the skip term below keeps the
        # un-dropped x_stalk (reference parity).
        z = self._apply_stalk_transform(self.dropout_layer(x_stalk))
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
