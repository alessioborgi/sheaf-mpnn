# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""Heterogeneous Bayesian Sheaf Neural Network (Het-BSNN) model.

Follows the BSNN architecture of Gillespie et al. (2025) with the het twist
applied where the sheaf actually gets learned: the variational sheaf learner
conditions the posterior over restriction maps on the het context (endpoint
features plus node/edge-type one-hots). The convolution layers themselves
consume sampled maps and are type-agnostic, so the deterministic
``sheaf_mpnn.bsnn`` layers are reused unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.bsnn.variational import (
    HetDiagonalVariationalSheafLearner,
    HetGeneralVariationalSheafLearner,
    HetOrthogonalVariationalSheafLearner,
    HetVariationalSheafLearner,
)
from sheaf_mpnn.bsnn.bsnn_layers import BaseBSNNConv
from sheaf_mpnn.bsnn.bsnn_model import BSNNVariant, reverse_edge_permutation


class HetBSNNModel(nn.Module):
    """Heterogeneous Bayesian Sheaf Neural Network for node-level prediction.

    Identical pipeline to ``BSNNModel`` -- encoder, one variational
    parameterization per forward pass, one i.i.d. sheaf sample per diffusion
    layer, decoder -- but the variational sheaf learner incorporates node/edge
    type information via the het context flags (use_features, use_node_types,
    use_edge_types). The KL term is exposed via :meth:`kl_loss` for the
    negative-ELBO training loop, exactly as in the homogeneous model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        variant: BSNNVariant = BSNNVariant.DIAGONAL,
        alpha: float = 1.0,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        add_eps: bool = False,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        learn_alpha: bool = True,
        ensemble_size: int = 3,
    ):
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if variant is BSNNVariant.ORTHOGONAL and stalk_dim < 2:
            raise ValueError("orthogonal BSNN requires stalk_dim >= 2")
        if variant is BSNNVariant.LOW_RANK:
            raise ValueError(
                "HetBSNN does not implement the low_rank variant: there is no "
                "heterogeneous low-rank variational learner; use the "
                "homogeneous BSNNModel, or diagonal/general/orthogonal here"
            )
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetBSNN layer")
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")

        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.variant = variant
        self.normalize_output = normalize_output
        self.jknet = jknet
        self.ensemble_size = ensemble_size
        context_dim = stalk_dim * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)

        het_kwargs: dict[str, Any] = dict(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
        )
        if variant is BSNNVariant.ORTHOGONAL:
            self.sheaf_learner: HetVariationalSheafLearner = (
                HetOrthogonalVariationalSheafLearner(
                    stalk_dim,
                    hidden_dim,
                    context_dim,
                    orth_strategy=orth_strategy,
                    clamp_val=clamp_val,
                    **het_kwargs,
                )
            )
        elif variant is BSNNVariant.GENERAL:
            self.sheaf_learner = HetGeneralVariationalSheafLearner(
                stalk_dim, hidden_dim, context_dim, **het_kwargs
            )
        else:
            self.sheaf_learner = HetDiagonalVariationalSheafLearner(
                stalk_dim, hidden_dim, context_dim, **het_kwargs
            )

        layer_class = variant.layer_class
        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    alpha=alpha,
                    add_eps=add_eps,
                    dropout=dropout,
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

        # KL of the most recent forward pass; zero-dim tensor for a clean
        # ``+`` in the loss even before the first forward.
        self._last_kl: torch.Tensor = torch.zeros(())

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.sheaf_learner.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BaseBSNNConv)
            layer.reset_parameters()
        if isinstance(self.decoder, nn.Linear):
            self.decoder.reset_parameters()

    def kl_loss(self) -> torch.Tensor:
        """KL divergence of the posterior from the prior, computed during the
        most recent forward pass (averaged over incident node-edge pairs).
        """
        return self._last_kl

    def forward(self, x, edge_index, node_types=None, edge_types=None):
        """Runs the encoder, sampled sheaf-diffusion layers, and decoder.

        The variational parameters are computed once from the het context;
        each layer receives an independent sheaf sample from the same
        posterior. The KL term of the ELBO is readable via :meth:`kl_loss`.

        Args:
            x (Tensor): Raw node features ``[N, in_channels]``.
            edge_index (LongTensor): Graph connectivity ``[2, E]``.
            node_types (LongTensor, optional): Node-type indices ``[N]``.
            edge_types (LongTensor, optional): Edge-type indices ``[E]``.

        Returns:
            Tensor: Node outputs ``[N, out_channels]``.
        """
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

        # Lift raw features to stalk space (reference: lin1 -> ELU -> dropout
        # before parameterizing the sheaf distribution).
        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))

        num_nodes = x_stalk.size(0)
        x_feat = x_stalk
        x_stalk = x_stalk.view(num_nodes, self.stalk_dim, self.hidden_dim)

        has_edges = edge_index.size(1) > 0
        if has_edges:
            # Variational parameters: computed once, shared by all layers.
            params = self.sheaf_learner(x_feat, edge_index, node_types, edge_types)
            self._last_kl = self.sheaf_learner.kl(params)
            rev = reverse_edge_permutation(edge_index, num_nodes)
        else:
            params, rev = None, None
            self._last_kl = x_feat.new_zeros(())

        layer_reps = []
        for layer in self.layers:
            if has_edges:
                assert params is not None and rev is not None
                # One i.i.d. sheaf per layer: row i holds the map of the
                # incident pair src_i ⊴ e_i; the reverse permutation gathers
                # the destination map of each directed edge.
                maps_src = self.sheaf_learner.sample(params)
                maps = torch.stack((maps_src[rev], maps_src))
            else:
                map_shape = (
                    (2, 0, self.stalk_dim)
                    if self.variant is BSNNVariant.DIAGONAL
                    else (2, 0, self.stalk_dim, self.stalk_dim)
                )
                maps = x_stalk.new_zeros(map_shape)
            x_stalk = layer(maps, x_stalk, edge_index)
            if self.jknet:
                h = x_stalk.reshape(num_nodes, -1)
                if self.normalize_output:
                    h = F.normalize(h, p=2, dim=-1)
                layer_reps.append(h)

        if self.jknet:
            x_out = self.jk(layer_reps)
        else:
            x_out = x_stalk.reshape(num_nodes, -1)

        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=-1)

        return self.decoder(x_out)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """Ensemble predictive probabilities over stochastic forward passes.

        Each pass samples a fresh collection of sheaves from the learned
        posterior; the per-pass softmax probabilities are returned stacked so
        callers can both average them (the ensemble prediction) and measure
        their spread (epistemic uncertainty).

        Args:
            x: Raw node features ``[N, in_channels]``.
            edge_index: Graph connectivity ``[2, E]``.
            node_types: Node-type indices ``[N]`` or None.
            edge_types: Edge-type indices ``[E]`` or None.
            num_samples: Number of forward passes; defaults to
                ``self.ensemble_size``.

        Returns:
            Tensor: Stacked probabilities ``[num_samples, N, C]``.
        """
        runs = num_samples if num_samples is not None else self.ensemble_size
        probs = [
            F.softmax(self(x, edge_index, node_types, edge_types), dim=-1)
            for _ in range(runs)
        ]
        return torch.stack(probs)


__all__ = ["HetBSNNModel"]
