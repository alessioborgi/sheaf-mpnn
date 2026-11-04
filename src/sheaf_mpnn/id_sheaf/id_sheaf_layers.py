import torch

from sheaf_mpnn.nsd.nsd_layers import BaseNSDConv
from sheaf_mpnn.utils import apply_diagonal_norm


class IdentitySheafConv(BaseNSDConv):
    """Identity Sheaf Network (ISN) from Hernandez Caralt et al. (2024).

    Removes learning parameters entirely from the Sheaf Laplacian, fixing
    all restriction maps cleanly to the Identity Matrix. The Laplacian is
    degree-normalized like every other family; with identity maps this is
    exactly the (D+1)^{-1/2} augmented graph Laplacian per stalk channel.
    """

    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E = edge_index.size(1)
        # Since F = I, F^T @ F = I. We output diagonal coefficients of 1.
        ones = torch.ones((E, self.d), dtype=x_feat.dtype, device=x_feat.device)
        return self._append_fixed_diag(ones, ones)

    def _apply_norm(
        self,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ):
        return apply_diagonal_norm(self_map, cross_map, edge_index, num_nodes)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        return self_map[:, :, None] * z_dst - cross_map[:, :, None] * z_src
