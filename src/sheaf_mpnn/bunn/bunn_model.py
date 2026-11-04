# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from sheaf_mpnn.bunn.bundle import PHI_KINDS, BundleMapLearner, PhiKind
from sheaf_mpnn.bunn.bunn_layers import (
    BuNNConv,
    BuNNSolver,
    Normalization,
    laplacian_eigendecomposition,
)


class BuNNModel(nn.Module):
    r"""End-to-end Bundle Neural Network (BuNN).

    BuNN (Bamberger et al., ICLR 2025) learns by *message diffusion* rather
    than message passing.  Each layer computes one orthogonal map per node
    (Eq. 1), applies the bundle-aware encoder in the synchronised frame
    (Eq. 2), evolves the features with the bundle heat kernel for a time
    :math:`t` (Eq. 3) -- which, the bundle being flat, reduces to the plain
    graph heat kernel by Lemma 3.1 -- and closes with a nonlinearity (Eq. 4).

    Two properties follow from that and motivate the architecture.  Every pair
    of nodes communicates in a *single* layer, since
    :math:`\mathcal{H}(t, u, v) > 0` everywhere, so there is no under-reaching
    and ``t`` tunes the scale at which distant nodes interact instead of depth
    doing it -- the paper's answer to over-squashing.  And the heat flow's
    fixed points obey :math:`O_v y_v = O_u y_u` rather than
    :math:`y_v = y_u`, so a non-trivial bundle keeps node-level information
    alive as depth grows -- its answer to over-smoothing.

    The wrapper keeps the family's shared skeleton: encode raw node features
    into the ``num_bundles * bundle_dim * hidden_dim`` bundle space, apply a
    stack of BuNN layers, decode the flat representation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bundle_dim: int = 2,
        hidden_dim: int = 16,
        num_bundles: int = 4,
        num_layers: int = 2,
        tau: float = 1.0,
        learn_tau: bool = False,
        solver: BuNNSolver = "taylor",
        max_degree: int | None = None,
        num_eigs: int | None = None,
        normalize: Normalization = "sym",
        bias: bool = True,
        phi: PhiKind = "mlp",
        num_phi_layers: int = 2,
        shared_phi: bool = False,
        pe_size: int = 0,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
    ):
        r"""Initializes a BuNN model for node-level prediction.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (e.g. classes).
            bundle_dim: Stalk dimension ``d`` of each bundle. ``2`` -- the
                paper's choice everywhere, since higher dimensions cost more
                without helping -- uses the direct O(2) angle parameterisation.
            hidden_dim: Vector-field channels per bundle. The node state has
                width ``num_bundles * bundle_dim * hidden_dim``.
            num_bundles: Number of parallel bundles ``b``.
            num_layers: Number of BuNN layers. Must be positive.
            tau: Diffusion time ``t``, the scale at which messages diffuse.
            learn_tau: Learn one diffusion time per bundle. Off by default;
                the reference turns it on for the spectral solver.
            solver: Heat-kernel evaluation -- ``"taylor"`` (Algorithm 1, cheap,
                for small ``t``) or ``"spectral"`` (Algorithm 2, exact for any
                ``t``, dense eigendecomposition).
            max_degree: Taylor truncation degree (``"taylor"`` only).
                ``None`` derives the smallest accurate degree from ``tau``; an
                explicit value too small for ``tau`` is rejected rather than
                diffusing with a divergent series.
            num_eigs: Truncate the spectral solver to this many low-frequency
                eigenpairs -- the paper's truncated spectral solution.
                ``None`` keeps the full spectrum and is exact.
            bias: Whether each layer's Eq. 2 encoder carries a bias.
            normalize: Laplacian normalisation, ``"sym"`` or ``"rw"``
                (``"spectral"`` requires ``"sym"``).
            phi: Bundle-map learner backbone -- a node-level ``"mlp"`` as in
                NSD, or the paper's ``"sum_gnn"``, which reads the graph
                structure and improved its results.
            num_phi_layers: Depth of the map learner.
            shared_phi: Share one map learner across layers (the paper's
                "shared phi"); otherwise each layer learns its own.
            pe_size: Width of the positional encodings ``P`` passed to
                ``forward`` and fed to :math:`\phi` alongside the features, as
                Eq. 1's ``phi(G, P, X, v)`` prescribes. Theorem 5.3's
                universality assumes these are injective, and they also keep
                deep stacks from drifting to a trivial bundle.
            orth_strategy: Orthogonal parameterisation for ``bundle_dim != 2``.
            clamp_val: Clamp on the orthogonal-transform parameters.
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on hidden states between layers.
            normalize_output: If ``True``, L2-normalise the representation
                before the decoder (Lv et al., 2021).
            jknet: If ``True``, concatenate every layer's hidden state before
                the decoder (Xu et al., 2018). Intended for link prediction.

        Raises:
            ValueError: On non-positive dimensions/layers or an unknown
                ``phi``.
        """
        super().__init__()
        if bundle_dim <= 0:
            raise ValueError("bundle_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_bundles <= 0:
            raise ValueError("num_bundles must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one BuNN layer")
        if phi not in PHI_KINDS:
            raise ValueError(f"phi must be one of {PHI_KINDS}, got {phi!r}")
        if pe_size < 0:
            raise ValueError("pe_size must be non-negative")

        self.bundle_dim = bundle_dim
        self.hidden_dim = hidden_dim
        self.num_bundles = num_bundles
        self.num_layers = num_layers
        self.solver: BuNNSolver = solver
        self.width = num_bundles * bundle_dim * hidden_dim
        self.pe_size = pe_size
        self.num_eigs = num_eigs
        self.normalize_output = normalize_output
        self.jknet = jknet

        self.input_dropout_layer = nn.Dropout(p=input_dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder = nn.Linear(in_channels, self.width)

        self.shared_map_learner = (
            BundleMapLearner(
                context_dim=self.width + pe_size,
                num_bundles=num_bundles,
                bundle_dim=bundle_dim,
                phi=phi,
                num_layers=num_phi_layers,
                orth_strategy=orth_strategy,
                clamp_val=clamp_val,
            )
            if shared_phi
            else None
        )

        self.layers = nn.ModuleList(
            [
                BuNNConv(
                    num_bundles=num_bundles,
                    bundle_dim=bundle_dim,
                    in_channels=hidden_dim,
                    tau=tau,
                    learn_tau=learn_tau,
                    solver=solver,
                    max_degree=max_degree,
                    num_eigs=num_eigs,
                    bias=bias,
                    context_dim=self.width,
                    pe_size=pe_size,
                    normalize=normalize,
                    phi=phi,
                    num_phi_layers=num_phi_layers,
                    orth_strategy=orth_strategy,
                    clamp_val=clamp_val,
                    external_maps=shared_phi,
                )
                for _ in range(num_layers)
            ]
        )

        decoder_in = (num_layers if jknet else 1) * self.width
        self.decoder = nn.Linear(decoder_in, out_channels)
        if jknet:
            self.jk = JumpingKnowledge(
                mode="cat", channels=self.width, num_layers=num_layers
            )

        # Per-graph eigendecomposition cache for the spectral solver: on a
        # transductive task every forward sees the same graph, so the dense
        # eigh runs once. A plain attribute, not a buffer -- it belongs to the
        # input graph, not to the model's parameters.
        #
        # It holds a single entry, which is the right shape for node-level
        # tasks but never hits under graph-level batching, where each batch is
        # a different disjoint union. Prefer solver="taylor" there: it is
        # sparse and carries no per-graph setup.
        self._eig_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        if self.shared_map_learner is not None:
            self.shared_map_learner.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, BuNNConv)
            layer.reset_parameters()
        self.decoder.reset_parameters()

    def _graph_eigs(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(eig_vals, eig_vecs)`` for this graph, cached across calls."""
        cached = self._eig_cache
        if (
            cached is not None
            and cached[0].shape == edge_index.shape
            and cached[0].device == edge_index.device
            and cached[2].size(0) == num_nodes
            and torch.equal(cached[0], edge_index)
        ):
            return cached[1], cached[2]
        eig_vals, eig_vecs = laplacian_eigendecomposition(
            edge_index, num_nodes, dtype=dtype, num_eigs=self.num_eigs
        )
        self._eig_cache = (edge_index.clone(), eig_vals, eig_vecs)
        return eig_vals, eig_vecs

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Runs the BuNN encoder, diffusion layers, and decoder.

        Args:
            x: Raw node features ``[num_nodes, in_channels]``.
            edge_index: Graph connectivity ``[2, num_edges]``, assumed
                undirected (both directions stored).
            pe: Positional encodings ``[num_nodes, pe_size]`` fed to
                :math:`\phi` alongside the features (Eq. 1). Required when the
                model was built with ``pe_size > 0``.

        Returns:
            Node outputs ``[num_nodes, out_channels]``.
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
        if self.pe_size and pe is None:
            raise ValueError(
                f"model was built with pe_size={self.pe_size}; pass pe to forward"
            )
        if pe is not None and pe.size(-1) != self.pe_size:
            raise ValueError(
                f"pe has width {pe.size(-1)} but pe_size is {self.pe_size}"
            )

        h = self.encoder(self.input_dropout_layer(x))

        eig_vals = eig_vecs = None
        if self.solver == "spectral":
            eig_vals, eig_vecs = self._graph_eigs(edge_index, h.size(0), h.dtype)

        layer_reps = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                h = self.dropout_layer(h)
            node_rep = None
            if self.shared_map_learner is not None:
                context = h if pe is None else torch.cat([h, pe], dim=-1)
                node_rep = self.shared_map_learner(context, edge_index)
            h = layer(h, h, edge_index, node_rep, eig_vecs, eig_vals, pe)
            # Eq. 4: the layer's only nonlinearity, applied after diffusion.
            h = F.elu(h)
            if self.jknet:
                layer_reps.append(
                    F.normalize(h, p=2, dim=-1) if self.normalize_output else h
                )

        if self.jknet:
            return self.decoder(self.jk(layer_reps))
        if self.normalize_output:
            h = F.normalize(h, p=2, dim=-1)
        return self.decoder(h)


__all__ = ["BuNNModel"]
