# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.dnsd.dnsd_layers import HetDNSDConv
from hetsheaf_mpnn.sheaf_adj.sheaf_adj_model import HetSheafAdjVariant

# As in the homogeneous family, the restriction-map family is orthogonal to the
# DNSD deltas (odd activation, gate, LayerNorm), so the het-SheafAdj enum is
# reused directly.
HetDNSDVariant = HetSheafAdjVariant


class HetDNSDModel(nn.Module):
    """Heterogeneous Deep Neural Sheaf Diffusion (HetDNSD) model.

    Same encoder-layers-decoder pipeline as
    :class:`~sheaf_mpnn.dnsd.dnsd_model.DNSDModel` (per-layer LayerNorm, odd
    activation and gate), but each layer is a
    :class:`~hetsheaf_mpnn.dnsd.dnsd_layers.HetDNSDConv` wrapping a
    ``residual=False`` het-SheafAdjacency layer, whose restriction-map MLP
    consumes the het context flags (``use_features``, ``use_node_types``,
    ``use_edge_types``, ``ensemble``).

    The encoder is shared across all node types (shared cochain projection),
    matching the existing heterogeneous experiment pipeline which flattens
    ``HeteroData`` via PyG's ``to_homogeneous()`` before passing it here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: HetDNSDVariant = HetDNSDVariant.GENERAL,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        ensemble: bool = False,
        add_self_loops: bool = False,
        orth_strategy: str = "cayley",
        rank: int = 1,
        add_eps: bool = False,
        odd: bool = False,
        gate: bool = False,
        layer_norm: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        learn_alpha: bool = True,
    ):
        """Initialise the HetDNSD model.

        Args:
            in_channels: Raw input feature dimension per node.
            out_channels: Output dimension (number of classes for node classification).
            stalk_dim: Stalk dimension d; restriction maps are d x d
                (or d x rank for LOW_RANK).
            hidden_dim: Hidden dimension h; cochain vectors are d x h.
            num_layers: Number of HetDNSD layers.
            variant: Which restriction-map parameterisation to use.
            alpha: Initial learnable step size of the adjacency term.
            num_node_types: Total number of node types (used when
                ``use_node_types=True``).
            num_edge_types: Total number of edge types (used when
                ``use_edge_types=True`` or ``ensemble=True``).
            use_features: Pass node features to the restriction-map MLP.
            use_node_types: Pass node-type embeddings to the restriction-map MLP.
            use_edge_types: Pass edge-type embeddings to the restriction-map MLP.
            ensemble: Use one restriction-map MLP per edge type.
            add_self_loops: Add self-loops to ``edge_index`` inside each layer.
            orth_strategy: Parameterisation for orthogonal maps
                (``"cayley"`` or ``"householder"``).
            rank: Rank for LOW_RANK variant.
            add_eps: Learnable per-stalk rescaling of the skip term (Remark 4).
            odd: Use ``tanh`` instead of the inherited ``ELU`` activation inside
                every layer's diffusion term.
            gate: Apply a learnable per-(node, stalk) sigmoid gate to every
                layer's diffusion term before the skip is added.
            layer_norm: Apply ``nn.LayerNorm`` to the full per-node embedding
                after every layer.
            input_dropout: Dropout probability applied to raw input features.
            dropout: Dropout probability applied between layers.
            normalize_output: L2-normalise cochain vectors before decoding.
            jknet: Use Jumping Knowledge (concatenation) to aggregate
                layer representations.
            learn_alpha: If False, alpha stays fixed at its init value.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetDNSD layer")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet
        self.layer_norm = layer_norm
        context_dim = stalk_dim * hidden_dim
        layer_class = variant.layer_class

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        extra_kwargs = variant.layer_kwargs.copy()
        if variant == HetDNSDVariant.ORTHOGONAL:
            extra_kwargs["orth_strategy"] = orth_strategy
        if variant == HetDNSDVariant.LOW_RANK:
            extra_kwargs["rank"] = rank

        def _make_layer() -> HetDNSDConv:
            sheaf_adj_layer = layer_class(
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
                add_self_loops=add_self_loops,
                add_eps=False,
                **extra_kwargs,
            )
            # Bare diffusion term only -- HetDNSDConv owns the single skip
            # connection (see sheaf_mpnn.dnsd.dnsd_layers.DNSDConv).
            sheaf_adj_layer.residual = False
            if odd:
                sheaf_adj_layer.sigma = nn.Tanh()
            return HetDNSDConv(
                sheaf_adj_layer,
                stalk_dim=stalk_dim,
                in_channels=hidden_dim,
                gate=gate,
                add_eps=add_eps,
            )

        self.layers = nn.ModuleList([_make_layer() for _ in range(num_layers)])
        if not learn_alpha:
            for layer in self.layers:
                assert isinstance(layer, HetDNSDConv)
                layer.sheaf_adj_layer.alpha.requires_grad_(False)

        self.norms = (
            nn.ModuleList([nn.LayerNorm(context_dim) for _ in range(num_layers)])
            if layer_norm
            else None
        )

        decoder_in = (num_layers if jknet else 1) * context_dim
        self.decoder = nn.Linear(decoder_in, out_channels)

        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=context_dim, num_layers=num_layers
            )

    def reset_parameters(self):
        """Reset all learnable parameters to their initial values."""
        self.encoder.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, HetDNSDConv)
            layer.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                assert isinstance(norm, nn.LayerNorm)
                norm.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs encoder, het-DNSD layers, and decoder.

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
        for i, layer in enumerate(self.layers):
            x_feat = self.dropout_layer(x_stalk.reshape(x_stalk.size(0), -1))
            x_stalk = layer(x_feat, x_stalk, edge_index, node_types, edge_types)
            if self.norms is not None:
                flat = self.norms[i](x_stalk.reshape(x_stalk.size(0), -1))
                x_stalk = flat.view(-1, self.stalk_dim, self.hidden_dim)
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


__all__ = ["HetDNSDVariant", "HetDNSDModel"]
