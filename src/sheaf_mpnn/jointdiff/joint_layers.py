import torch
import torch.nn.init as init
from torch import nn

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.utils import apply_general_norm


class JointDiffusionSheafConv(BaseSheafConv):
    """Joint Diffusion Sheaf Convolution matching OpinionDynamicsFixedSheaf.

    Evolves the restriction maps along with the features using the ODE:
    F_src = F_src - beta * tanh((F_src @ x_src - F_dst @ x_dst) @ x_src^T)
    F_dst = F_dst - beta * tanh((F_dst @ x_dst - F_src @ x_src) @ x_dst^T)

    Initial maps F_src, F_dst are learned from concatenated endpoint features
    via a small MLP, and then transformed via learnable matrices Wf1 and Wf2.
    Only the first layer of a stack owns the initializer (``learn_init_maps``);
    later layers receive the evolved maps via ``prev_maps``.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        beta: float = 1.0,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = False,
        add_eps: bool = True,
        dropout: float = 0.0,
        add_lp: bool = False,
        add_hp: bool = False,
        learn_init_maps: bool = True,
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
        self.beta = beta
        self.dropout_layer = nn.Dropout(dropout)

        if self.add_eps:
            self.eps = nn.Parameter(torch.zeros(self.stalk_dim))

        # Learned initializer for F_src, F_dst from concatenated [x_src, x_dst].
        # Only the stack's first layer needs it: later layers always receive
        # the evolved maps via ``prev_maps``, so giving each of them a
        # generator would create parameters that never see a gradient.
        if learn_init_maps:
            init_in_dim = 2 * self.d * self.in_channels
            init_out_dim = 2 * self.d * self.d
            self.init_map_generator = nn.Sequential(
                nn.Linear(init_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, init_out_dim),
            )
        else:
            self.init_map_generator = None

        # Learnable transformation matrices for the restriction maps
        self.Wf1 = nn.Parameter(torch.empty(self.d, self.d))
        self.Wf2 = nn.Parameter(torch.empty(self.d, self.d))

        self.reset_parameters()

    def reset_parameters(self):
        # W1 identity / W2 orthogonal (reference parity) via the base class.
        super().reset_parameters()
        init.eye_(self.Wf1)
        init.eye_(self.Wf2)
        if self.init_map_generator is not None:
            for m in self.init_map_generator:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.zeros_(m.bias)

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
            learned_dim=self.d,
            detach_degree=True,
        )

    def _transform_map(self, F: torch.Tensor) -> torch.Tensor:
        """Applies the bilinear transformation Wf1 @ F @ Wf2."""
        return torch.matmul(torch.matmul(self.Wf1, F), self.Wf2)

    def _init_maps(self, x_feat_reshaped, src_idx, dst_idx):
        """Learn initial F_src, F_dst from concatenated endpoint features."""
        if self.init_map_generator is None:
            raise RuntimeError(
                "This layer has learn_init_maps=False and received no prev_maps;"
                " only the first layer of a stack may initialize the maps."
            )
        E = src_idx.size(0)
        x_src_vec = x_feat_reshaped[src_idx]  # [E, d, f]
        x_dst_vec = x_feat_reshaped[dst_idx]  # [E, d, f]

        concat_feat = torch.cat(
            [x_src_vec.reshape(E, -1), x_dst_vec.reshape(E, -1)], dim=-1
        )  # [E, 2*d*f]

        raw = self.init_map_generator(concat_feat)  # [E, 2*d*d]
        F_src, F_dst = raw.chunk(2, dim=-1)
        F_src = F_src.view(E, self.d, self.d)
        F_dst = F_dst.view(E, self.d, self.d)

        return F_src, F_dst

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
        prev_maps: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        num_nodes = x_feat.size(0)
        src_idx, dst_idx = edge_index

        # 1. Initialize (learned) or load previous maps [E, d, d]
        if prev_maps is None:
            # Reshape features to [num_nodes, final_d, f], keep the learned dims.
            x_feat_reshaped = x_feat.view(num_nodes, self.stalk_dim, self.in_channels)[
                :, : self.d, :
            ]
            F_src, F_dst = self._init_maps(x_feat_reshaped, src_idx, dst_idx)
        else:
            F_src, F_dst = prev_maps

        # 2. Joint Diffusion ODE step
        if self.beta > 0:
            x_stalk_d = x_stalk[:, : self.d, :]
            x_src_vec = x_stalk_d[src_idx]  # [E, d, hidden_dim]
            x_dst_vec = x_stalk_d[dst_idx]  # [E, d, hidden_dim]

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

        self_map, cross_map = self._append_fixed_block(self_map, cross_map)
        norm_self, norm_cross = self._apply_norm(
            self_map, cross_map, edge_index, num_nodes
        )

        # 3. Propagate stalk features; dropout hits only the diffusion input,
        # the skip term below keeps the un-dropped x_stalk (reference parity).
        z = self._apply_stalk_transform(self.dropout_layer(x_stalk))
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
