import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.jointdiff.joint_layers import JointDiffusionSheafConv
from sheaf_mpnn.jointdiff.rot_inv_layers import RotationInvariantSheafConv


class JointDiffusionModel(nn.Module):
    """End-to-end Joint Diffusion Sheaf Neural Network (JdSNN) model.

    The restriction maps are jointly diffused together with node features
    using the discrete ODE step evaluated locally per-edge.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        beta: float = 1.0,
        alpha: float = 1.0,
        add_self_loops: bool = False,
        add_eps: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        second_linear: bool = False,
        learn_alpha: bool = True,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one layer")

        self.stalk_dim = stalk_dim
        self.final_d = stalk_dim + int(add_lp) + int(add_hp)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = self.final_d * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        # Only the first layer learns the map initializer; later layers evolve
        # the maps received via prev_maps.
        self.layers = nn.ModuleList(
            [
                JointDiffusionSheafConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    beta=beta,
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    dropout=dropout,
                    add_lp=add_lp,
                    add_hp=add_hp,
                    learn_init_maps=(i == 0),
                )
                for i in range(num_layers)
            ]
        )
        if not learn_alpha:
            # Bodnar-exact update: the diffusion step size stays fixed.
            for layer in self.layers:
                layer.alpha.requires_grad_(False)

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, JointDiffusionSheafConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        if x.dim() != 2 or x.size(1) != self.encoder.in_features:
            raise ValueError(
                f"x must have shape [num_nodes, {self.encoder.in_features}],"
                f" got {tuple(x.shape)}"
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges],"
                f" got {tuple(edge_index.shape)}"
            )

        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.final_d, self.hidden_dim)

        layer_reps = []

        # prev_maps: Tuple of [E, d, d] left and right restriction maps from
        # the previous layer's ODE step.
        prev_maps = None

        for i, layer in enumerate(self.layers):
            x_feat = x_stalk.reshape(x_stalk.size(0), -1)
            if i > 0:
                x_feat = self.dropout_layer(x_feat)

            x_stalk, prev_maps = layer(x_feat, x_stalk, edge_index, prev_maps=prev_maps)

            if self.jknet:
                h = x_stalk.reshape(x_stalk.size(0), -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk.reshape(x_stalk.size(0), -1)

        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)


class RotationInvariantModel(nn.Module):
    """End-to-end Rotation Invariant Sheaf Neural Network (RiSNN) model.

    Each layer computes restriction maps from rotation-invariant outer products
    of the node features across the edge, iteratively passing the maps forward
    when ``time_dep=True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        alpha: float = 1.0,
        add_self_loops: bool = False,
        add_eps: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        second_linear: bool = False,
        time_dep: bool = True,
        learn_alpha: bool = True,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one layer")

        self.stalk_dim = stalk_dim
        self.final_d = stalk_dim + int(add_lp) + int(add_hp)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.normalize_output = normalize_output
        self.jknet = jknet
        self.time_dep = time_dep
        context_dim = self.final_d * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        self.layers = nn.ModuleList(
            [
                RotationInvariantSheafConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    add_eps=add_eps,
                    dropout=dropout,
                    add_lp=add_lp,
                    add_hp=add_hp,
                    time_dep=time_dep,
                )
                for _ in range(num_layers)
            ]
        )
        if not learn_alpha:
            # Bodnar-exact update: the diffusion step size stays fixed.
            for layer in self.layers:
                layer.alpha.requires_grad_(False)

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, RotationInvariantSheafConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index):
        if x.dim() != 2 or x.size(1) != self.encoder.in_features:
            raise ValueError(
                f"x must have shape [num_nodes, {self.encoder.in_features}],"
                f" got {tuple(x.shape)}"
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges],"
                f" got {tuple(edge_index.shape)}"
            )

        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)
        x_stalk = x_stalk.view(-1, self.final_d, self.hidden_dim)

        layer_reps = []

        # prev_maps: [E, d, d] restriction maps from the previous layer, used
        # in time-dependent mode.
        prev_maps: torch.Tensor | None = None

        for i, layer in enumerate(self.layers):
            x_feat = x_stalk.reshape(x_stalk.size(0), -1)
            if i > 0:
                x_feat = self.dropout_layer(x_feat)

            # Unpack the updated stalk features and the maps for the next layer
            x_stalk, prev_maps = layer(x_feat, x_stalk, edge_index, prev_maps=prev_maps)

            if self.jknet:
                h = x_stalk.reshape(x_stalk.size(0), -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk.reshape(x_stalk.size(0), -1)

        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)
