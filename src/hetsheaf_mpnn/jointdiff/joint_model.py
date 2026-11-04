# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.jointdiff.joint_layers import HetJointDiffusionSheafConv
from hetsheaf_mpnn.jointdiff.rot_inv_layers import HetRotationInvariantSheafConv


class HetJointDiffusionModel(nn.Module):
    """Heterogeneous Joint Diffusion Sheaf Neural Network (Het-JdSNN) model.

    Identical encoder-layers-decoder pipeline to HetNSDModel, but the
    restriction maps are jointly diffused together with node features using
    the discrete ODE step evaluated locally per-edge. The first layer's map
    initializer consumes the het context (features and node/edge-type
    one-hots) controlled by the standard het flags.
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
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_eps: bool = False,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetJointDiffusion layer")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.normalize_output = normalize_output
        self.jknet = jknet
        context_dim = stalk_dim * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        # Only the first layer learns the map initializer; later layers evolve
        # the maps received via prev_maps.
        self.layers = nn.ModuleList(
            [
                HetJointDiffusionSheafConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    beta=beta,
                    alpha=alpha,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    add_eps=add_eps,
                    learn_init_maps=(i == 0),
                )
                for i in range(num_layers)
            ]
        )

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, HetJointDiffusionSheafConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, het joint-diffusion layers, and decoder.

        Args:
            x (Tensor): Raw node features [N, in_channels].
            edge_index (LongTensor): Graph connectivity [2, E].
            node_types (LongTensor, optional): Node-type indices [N].
            edge_types (LongTensor, optional): Edge-type indices [E].

        Returns:
            Tensor: Node outputs [N, out_channels].
        """
        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )

        layer_reps = []

        # prev_maps: Tuple of [E, d, d] left and right restriction maps from
        # the previous layer's ODE step.
        prev_maps = None

        for layer in self.layers:
            x_feat = self.dropout_layer(x_stalk.reshape(x_stalk.size(0), -1))
            x_stalk, prev_maps = layer(
                x_feat, x_stalk, edge_index, node_types, edge_types, prev_maps=prev_maps
            )
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


class HetRotationInvariantModel(nn.Module):
    """Heterogeneous Rotation Invariant Sheaf Neural Network (Het-RiSNN) model.

    Each layer computes restriction maps from rotation-invariant outer
    products of the node features across the edge, extended with node/edge
    type one-hots, iteratively passing the maps forward when ``time_dep=True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_eps: bool = False,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        time_dep: bool = True,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetRotationInvariant layer")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.normalize_output = normalize_output
        self.jknet = jknet
        self.time_dep = time_dep
        context_dim = stalk_dim * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        self.layers = nn.ModuleList(
            [
                HetRotationInvariantSheafConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    alpha=alpha,
                    num_node_types=num_node_types,
                    num_edge_types=num_edge_types,
                    use_features=use_features,
                    use_node_types=use_node_types,
                    use_edge_types=use_edge_types,
                    ensemble=ensemble,
                    context_dim=context_dim,
                    add_eps=add_eps,
                    time_dep=time_dep,
                )
                for _ in range(num_layers)
            ]
        )

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, HetRotationInvariantSheafConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, het rotation-invariant layers, and decoder.

        Args:
            x (Tensor): Raw node features [N, in_channels].
            edge_index (LongTensor): Graph connectivity [2, E].
            node_types (LongTensor, optional): Node-type indices [N].
            edge_types (LongTensor, optional): Edge-type indices [E].

        Returns:
            Tensor: Node outputs [N, out_channels].
        """
        x_stalk = self.encoder(self.input_dropout_layer(x)).view(
            -1, self.stalk_dim, self.hidden_dim
        )

        layer_reps = []

        # prev_maps: [E, d, d] restriction maps from the previous layer, used
        # in time-dependent mode.
        prev_maps: torch.Tensor | None = None

        for layer in self.layers:
            x_feat = self.dropout_layer(x_stalk.reshape(x_stalk.size(0), -1))
            x_stalk, prev_maps = layer(
                x_feat, x_stalk, edge_index, node_types, edge_types, prev_maps=prev_maps
            )
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


__all__ = ["HetJointDiffusionModel", "HetRotationInvariantModel"]
