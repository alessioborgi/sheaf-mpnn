# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""End-to-end Bayesian Sheaf Neural Network (BSNN) model.

Implements the architecture of Gillespie et al. (2025, arXiv:2410.09590,
Figure 2 / Algorithm 1) on top of the MPSNN formulation of this library:

1. An encoder lifts raw node features into stalk features (same pipeline as
   ``NSDModel``: dropout -> linear -> ELU -> dropout [-> second linear]).
2. A **variational sheaf learner** maps each incident node-edge pair's
   concatenated features to the parameters of a distribution over restriction
   maps -- computed **once** per forward pass.
3. One i.i.d. sheaf sample is drawn **per diffusion layer**; each
   ``*BSNNConv`` layer applies one sheaf-diffusion update with its sample.
4. A linear decoder maps the final stalk representation to class logits.

The KL divergence of the variational posterior from the prior is computed in
the same forward pass and exposed via :meth:`BSNNModel.kl_loss` so the
training loop can assemble the negative ELBO
:math:`\mathcal{L} = \mathrm{CE} + \lambda\,\mathrm{KL}` (paper Eq. 19/20,
with :math:`\lambda` cyclically annealed).

At test time, :meth:`BSNNModel.predict_proba` averages softmax probabilities
over an ensemble of stochastic forward passes (paper Section 2.5).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.bsnn.bsnn_layers import (
    BaseBSNNConv,
    DiagonalBSNNConv,
    GeneralBSNNConv,
    LowRankBSNNConv,
    OrthogonalBSNNConv,
)
from sheaf_mpnn.bsnn.variational import (
    BaseVariationalSheafLearner,
    DiagonalVariationalSheafLearner,
    GeneralVariationalSheafLearner,
    LowRankVariationalSheafLearner,
    OrthogonalVariationalSheafLearner,
)


class BSNNVariant(Enum):
    DIAGONAL = auto()
    GENERAL = auto()
    LOW_RANK = auto()
    ORTHOGONAL = auto()

    @property
    def layer_class(self) -> type[BaseBSNNConv]:
        mapping = {
            BSNNVariant.DIAGONAL: DiagonalBSNNConv,
            BSNNVariant.GENERAL: GeneralBSNNConv,
            BSNNVariant.LOW_RANK: LowRankBSNNConv,
            BSNNVariant.ORTHOGONAL: OrthogonalBSNNConv,
        }
        return mapping[self]


def reverse_edge_permutation(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Index of the reverse directed edge ``(v, u)`` for every edge ``(u, v)``.

    Sampled restriction maps are indexed per *incident node-edge pair*: row
    ``i`` of the sample array holds the map of ``src_i`` for edge
    ``{src_i, dst_i}``. The destination map of edge ``i`` is then the source
    map of its reverse edge, so gathering with this permutation guarantees
    every incident pair is sampled exactly once and used consistently in both
    orientations -- keeping the sampled sheaf Laplacian symmetric.

    Directed leftovers without a reverse edge fall back to their own index
    (the two incident maps of that edge coincide).

    Args:
        edge_index: Graph connectivity ``[2, E]``.
        num_nodes: Number of nodes (for collision-free edge keys).

    Returns:
        Permutation tensor ``[E]`` such that ``edge_index[:, perm[i]]`` equals
        ``edge_index[[1, 0], i]`` wherever the reverse edge exists.
    """
    src, dst = edge_index
    key_fwd = src * num_nodes + dst
    key_bwd = dst * num_nodes + src
    order = torch.argsort(key_fwd)
    pos = torch.searchsorted(key_fwd[order], key_bwd)
    pos = pos.clamp(max=max(order.numel() - 1, 0))
    rev = order[pos] if order.numel() else pos
    own = torch.arange(src.numel(), device=edge_index.device)
    matched = key_fwd[rev] == key_bwd if order.numel() else torch.zeros_like(own).bool()
    return torch.where(matched, rev, own)


class BSNNModel(nn.Module):
    """End-to-end Bayesian Sheaf Neural Network for node-level prediction.

    The wrapper lifts raw node features into stalk features, parameterizes a
    variational distribution over cellular sheaves, applies a stack of BSNN
    convolution layers -- each with its own sheaf sample -- and decodes the
    flattened stalk representation to the requested output dimension.
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
        rank: int = 1,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        add_eps: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
        add_lp: bool = False,
        add_hp: bool = False,
        sparse_learner: bool = False,
        second_linear: bool = False,
        learn_alpha: bool = True,
        ensemble_size: int = 3,
    ):
        """Initializes a BSNN model.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (num classes).
            stalk_dim: Number of learned stalk dimensions ``d``. For the
                ``ORTHOGONAL`` variant with ``d`` in {2, 3} the KL divergence
                uses the closed forms of the paper's Proposition 3; larger
                ``d`` uses a Monte Carlo estimate.
            hidden_dim: Feature dimension inside each stalk channel (``f``).
            num_layers: Number of BSNN convolution layers; one i.i.d. sheaf
                is sampled per layer.
            variant: Restriction-map family: ``DIAGONAL``, ``GENERAL``, and
                ``LOW_RANK`` use Gaussian posteriors with standard normal
                priors (``LOW_RANK`` over the factors of ``F = A B^T``, so
                every sampled map has rank at most ``rank``); ``ORTHOGONAL``
                uses Cayley distributions on SO(d) with a uniform prior.
            alpha: Initial diffusion step size per layer.
            rank: Rank bound of each restriction map for the ``LOW_RANK``
                variant (ignored otherwise).
            orth_strategy: Parameterization of the mean rotation for the
                ``ORTHOGONAL`` variant (ignored otherwise).
            clamp_val: Clamp for Cayley mean-map parameters.
            add_eps: Learnable per-stalk skip rescaling (as in NSD).
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on stalk features between layers.
            normalize_output: L2-normalise the representation before the
                decoder.
            jknet: Concatenate hidden states across layers before decoding.
            add_lp: Appends one fixed signless-Laplacian stalk channel.
            add_hp: Appends one fixed standard-Laplacian stalk channel.
            sparse_learner: Stalk-summed learner input (reference
                ``BayesConcatSheafLearnerVariant``).
            second_linear: Adds a second encoder linear after ELU+dropout.
            learn_alpha: If ``False``, alpha stays fixed at its init value.
            ensemble_size: Default number of stochastic forward passes
                averaged by :meth:`predict_proba`.
        """
        super().__init__()
        if stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        if variant is BSNNVariant.ORTHOGONAL and stalk_dim < 2:
            raise ValueError("orthogonal BSNN requires stalk_dim >= 2")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one BSNN layer")
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")

        self.stalk_dim = stalk_dim
        self.final_d = stalk_dim + int(add_lp) + int(add_hp)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.variant = variant
        self.rank = rank
        self.normalize_output = normalize_output
        self.jknet = jknet
        self.ensemble_size = ensemble_size
        context_dim = self.final_d * hidden_dim

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, context_dim)
        self.encoder2 = nn.Linear(context_dim, context_dim) if second_linear else None

        if variant is BSNNVariant.ORTHOGONAL:
            self.sheaf_learner: BaseVariationalSheafLearner = (
                OrthogonalVariationalSheafLearner(
                    stalk_dim,
                    hidden_dim,
                    context_dim,
                    sparse=sparse_learner,
                    orth_strategy=orth_strategy,
                    clamp_val=clamp_val,
                )
            )
        elif variant is BSNNVariant.GENERAL:
            self.sheaf_learner = GeneralVariationalSheafLearner(
                stalk_dim, hidden_dim, context_dim, sparse=sparse_learner
            )
        elif variant is BSNNVariant.LOW_RANK:
            self.sheaf_learner = LowRankVariationalSheafLearner(
                stalk_dim, hidden_dim, context_dim, sparse=sparse_learner, rank=rank
            )
        else:
            self.sheaf_learner = DiagonalVariationalSheafLearner(
                stalk_dim, hidden_dim, context_dim, sparse=sparse_learner
            )

        layer_class = variant.layer_class
        # Only LowRankBSNNConv reads the rank bound of its factored maps.
        layer_kwargs: dict[str, Any] = (
            {"rank": rank} if variant is BSNNVariant.LOW_RANK else {}
        )
        self.layers = nn.ModuleList(
            [
                layer_class(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    alpha=alpha,
                    add_eps=add_eps,
                    dropout=dropout,
                    add_lp=add_lp,
                    add_hp=add_hp,
                    **layer_kwargs,
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
        if self.encoder2 is not None:
            self.encoder2.reset_parameters()
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

    def forward(self, x, edge_index):
        """Runs the encoder, sampled sheaf-diffusion layers, and decoder.

        The variational parameters are computed once; each layer receives an
        independent sheaf sample from the same posterior. The KL term of the
        ELBO is stored and readable via :meth:`kl_loss`.

        Args:
            x (torch.Tensor): Raw node features ``[num_nodes, in_channels]``.
            edge_index (torch.Tensor): Graph connectivity ``[2, num_edges]``.

        Returns:
            torch.Tensor: Node outputs ``[num_nodes, out_channels]``.
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
        # [-> lin12] before parameterizing the sheaf distribution).
        x_stalk = self.encoder(self.input_dropout_layer(x))
        x_stalk = self.dropout_layer(F.elu(x_stalk))
        if self.encoder2 is not None:
            x_stalk = self.encoder2(x_stalk)

        num_nodes = x_stalk.size(0)
        x_feat = x_stalk
        x_stalk = x_stalk.view(num_nodes, self.final_d, self.hidden_dim)

        has_edges = edge_index.size(1) > 0
        if has_edges:
            # Variational parameters: computed once, shared by all layers.
            params = self.sheaf_learner(x_feat, edge_index)
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
                if self.variant is BSNNVariant.DIAGONAL:
                    map_shape = (2, 0, self.stalk_dim)
                elif self.variant is BSNNVariant.LOW_RANK:
                    map_shape = (2, 0, self.stalk_dim, 2 * self.rank)
                else:
                    map_shape = (2, 0, self.stalk_dim, self.stalk_dim)
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
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """Ensemble predictive probabilities over stochastic forward passes.

        Each pass samples a fresh collection of sheaves from the learned
        posterior; the per-pass softmax probabilities are returned stacked so
        callers can both average them (the ensemble prediction) and measure
        their spread (epistemic uncertainty).

        Args:
            x: Raw node features ``[num_nodes, in_channels]``.
            edge_index: Graph connectivity ``[2, num_edges]``.
            num_samples: Number of forward passes; defaults to
                ``self.ensemble_size``.

        Returns:
            torch.Tensor: Stacked probabilities ``[num_samples, N, C]``.
        """
        runs = num_samples if num_samples is not None else self.ensemble_size
        probs = [F.softmax(self(x, edge_index), dim=-1) for _ in range(runs)]
        return torch.stack(probs)


__all__ = ["BSNNModel", "BSNNVariant", "reverse_edge_permutation"]
