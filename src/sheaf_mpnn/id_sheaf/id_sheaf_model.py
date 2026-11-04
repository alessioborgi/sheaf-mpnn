import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.id_sheaf.id_sheaf_layers import IdentitySheafConv


class IdentitySheafModel(nn.Module):
    """End-to-end Identity Sheaf Network (ISN) model."""

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

        self.layers = nn.ModuleList(
            [
                IdentitySheafConv(
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
                )
                for _ in range(num_layers)
            ]
        )
        if not learn_alpha:
            for layer in self.layers:
                layer.alpha.requires_grad_(False)

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )
        # Initialize layer parameters after creation
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, IdentitySheafConv)
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
        for i, layer in enumerate(self.layers):
            x_feat = x_stalk.reshape(x_stalk.size(0), -1)
            if i > 0:
                x_feat = self.dropout_layer(x_feat)
            x_stalk = layer(x_feat, x_stalk, edge_index)
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
