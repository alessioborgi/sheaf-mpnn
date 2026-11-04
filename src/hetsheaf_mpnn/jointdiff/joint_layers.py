# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
import torch.nn.init as init
from torch import nn

from hetsheaf_mpnn.base_conv import BaseHetSheafConv
from sheaf_mpnn.utils import apply_general_norm


class HetJointDiffusionSheafConv(BaseHetSheafConv):
    """Heterogeneous Joint Diffusion Sheaf Convolution.

    Evolves the restriction maps along with the features using the ODE:
    F_src = F_src - beta * tanh((F_src @ x_src - F_dst @ x_dst) @ x_src^T)
    F_dst = F_dst - beta * tanh((F_dst @ x_dst - F_src @ x_src) @ x_dst^T)

    The initial maps are learned from the het edge context (endpoint features
    plus optional node/edge-type one-hots) via the standard het map-generator
    MLP; in ensemble mode one MLP per edge type initializes the maps. Only the
    first layer of a stack owns the initializer (``learn_init_maps``); later
    layers receive the evolved maps via ``prev_maps``, where the type
    information persists through the ODE evolution.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        beta: float = 1.0,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        context_dim: int | None = None,
        add_eps: bool = False,
        learn_init_maps: bool = True,
    ):
        super().__init__(
            stalk_dim,
            [in_channels],
            hidden_dim,
            context_dim,
            False,  # add_self_loops: joint layers never augment the graph.
            num_node_types,
            num_edge_types,
            use_features,
            use_node_types,
            use_edge_types,
            ensemble,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = beta
        self.add_eps = add_eps
        if add_eps:
            self.eps = nn.Parameter(torch.zeros(stalk_dim))

        # Only the stack's first layer needs the initializer: later layers
        # always receive the evolved maps via ``prev_maps``, so giving each of
        # them a generator would create parameters that never see a gradient.
        self.learn_init_maps = learn_init_maps
        if learn_init_maps:
            self._make_map_generators(hidden_dim, stalk_dim * stalk_dim)

        # Learnable transformation matrices for the restriction maps
        self.Wf1 = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.Wf2 = nn.Parameter(torch.empty(stalk_dim, stalk_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # W1/W2 Xavier plus near-zero map generators via the het base.
        super().reset_parameters()
        init.eye_(self.Wf1)
        init.eye_(self.Wf2)

    def _apply_norm(self, self_map, cross_map, edge_index, num_nodes):
        # The maps feeding the degree evolve through the ODE step; detaching
        # the degree keeps that evolution out of the normalization gradient.
        return apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            self.stalk_dim,
            self.training,
            detach_degree=True,
        )

    def _transform_map(self, F: torch.Tensor) -> torch.Tensor:
        """Applies the bilinear transformation Wf1 @ F @ Wf2."""
        return torch.matmul(torch.matmul(self.Wf1, F), self.Wf2)

    def _init_maps(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None,
        edge_types: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Learn initial F_src, F_dst from the het edge context."""
        if not self.learn_init_maps:
            raise RuntimeError(
                "This layer has learn_init_maps=False and received no prev_maps;"
                " only the first layer of a stack may initialize the maps."
            )
        if self.ensemble:
            raw = self._ensemble_forward(x_feat, edge_index, node_types, edge_types)
        else:
            inp = self._build_context_pair(x_feat, edge_index, node_types, edge_types)
            raw = self.map_generator(inp)
        # First E rows are the dst-perspective, last E the src-perspective.
        F_dst, F_src = (
            w.view(-1, self.stalk_dim, self.stalk_dim) for w in raw.chunk(2, dim=0)
        )
        return F_src, F_dst

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        prev_maps: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Applies one het joint-diffusion step.

        Args:
            x_feat:     [N, context_dim]             Flattened stalk (map context).
            x_stalk:    [N, stalk_dim, in_channels]  Structured stalk to update.
            edge_index: [2, E]                       Graph connectivity.
            node_types: [N] LongTensor or None       Node-type indices.
            edge_types: [E] LongTensor or None       Edge-type indices.
            prev_maps:  Optional (F_src, F_dst) evolved by the previous layer.

        Returns:
            (updated stalk [N, stalk_dim, in_channels], (F_src, F_dst)).
        """
        num_nodes = x_stalk.size(0)
        src_idx, dst_idx = edge_index

        # 1. Initialize (learned) or load previous maps [E, d, d]
        if prev_maps is None:
            F_src, F_dst = self._init_maps(x_feat, edge_index, node_types, edge_types)
        else:
            F_src, F_dst = prev_maps

        # 2. Joint Diffusion ODE step
        if self.beta > 0:
            x_src_vec = x_stalk[src_idx]  # [E, d, in_channels]
            x_dst_vec = x_stalk[dst_idx]  # [E, d, in_channels]

            diff = torch.bmm(self._transform_map(F_src), x_src_vec) - torch.bmm(
                self._transform_map(F_dst), x_dst_vec
            )
            F_src_next = F_src - self.beta * torch.tanh(
                torch.bmm(diff, x_src_vec.transpose(-2, -1))
            )
            F_dst_next = F_dst - self.beta * torch.tanh(
                torch.bmm(diff, -x_dst_vec.transpose(-2, -1))
            )
            F_src, F_dst = F_src_next, F_dst_next

        self_map = torch.bmm(F_dst.transpose(1, 2), F_dst)
        cross_map = torch.bmm(F_dst.transpose(1, 2), F_src)
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        # 3. Propagate stalk features.
        z = self._apply_stalk_transform(x_stalk)
        z_src, z_dst = z[src_idx], z[dst_idx]
        out = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            z_src=z_src,
            z_dst=z_dst,
            self_map=norm_self,
            cross_map=norm_cross,
            size=(num_nodes, num_nodes),
        )

        skip = (
            (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
            if self.add_eps
            else x_stalk
        )
        return skip - self.alpha * self.sigma(out), (F_src, F_dst)


__all__ = ["HetJointDiffusionSheafConv"]
