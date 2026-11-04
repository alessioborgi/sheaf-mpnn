# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous Bundle Neural Network (HetBuNN) model."""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.models import JumpingKnowledge

from hetsheaf_mpnn.bunn.bundle import HetBundleMapLearner
from hetsheaf_mpnn.bunn.bunn_layers import HetBuNNConv
from sheaf_mpnn.bunn.bundle import PHI_KINDS, PhiKind
from sheaf_mpnn.bunn.bunn_layers import (
    BuNNSolver,
    Normalization,
    laplacian_eigendecomposition,
)


class HetBuNNModel(nn.Module):
    r"""Heterogeneous Bundle Neural Network for node-level prediction.

    Same encoder -> BuNN layers -> decoder pipeline as
    :class:`~sheaf_mpnn.bunn.bunn_model.BuNNModel`, with the het context
    injected where the bundle is learned: every map learner :math:`\phi` (one
    per layer, or the shared one) is a
    :class:`~hetsheaf_mpnn.bunn.bundle.HetBundleMapLearner`. It reads
    node-type one-hots (``use_node_types``) and, with ``phi="sum_gnn"``, edge
    types through relation-specific neighbour weights (``use_edge_types``).
    ``use_features=False`` drops the node features from its input; with an MLP
    :math:`\phi` every node of one type then shares a frame, and the transport
    between two types is a learned per-type-pair rotation.

    Node types also do part of the job the paper gives positional encodings:
    :math:`\phi` did not produce them, so a deep stack that flattens the
    features cannot flatten them, and nodes of different types keep different
    frames. Unlike injective encodings they leave nodes of one type free to
    collapse together.

    The diffusion itself is type-blind (see
    :class:`~hetsheaf_mpnn.bunn.bunn_layers.HetBuNNConv`), there is no
    per-edge-type ``ensemble`` mode (maps are per node, not per edge), and the
    encoder is shared across node types, matching the het pipeline that
    flattens ``HeteroData`` with ``to_homogeneous()``.
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
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = False,
        jknet: bool = False,
    ):
        """Initializes a HetBuNN model.

        Args:
            in_channels: Number of raw input features per node.
            out_channels: Number of output channels per node (e.g. classes).
            bundle_dim: Stalk dimension ``d`` of each bundle (``2`` uses the
                direct O(2) angle parameterisation).
            hidden_dim: Vector-field channels per bundle. The node state has
                width ``num_bundles * bundle_dim * hidden_dim``.
            num_bundles: Number of parallel bundles ``b``.
            num_layers: Number of BuNN layers. Must be positive.
            tau: Diffusion time ``t``.
            learn_tau: Learn one diffusion time per bundle.
            solver: ``"taylor"`` (Algorithm 1) or ``"spectral"`` (Algorithm 2).
            max_degree: Taylor truncation degree; ``None`` derives it from
                ``tau``.
            num_eigs: Spectral truncation; ``None`` keeps the full spectrum.
            normalize: Laplacian normalisation, ``"sym"`` or ``"rw"``.
            bias: Whether each layer's Eq. 2 encoder carries a bias.
            phi: Map-learner backbone, ``"mlp"`` or ``"sum_gnn"``;
                ``use_edge_types`` needs ``"sum_gnn"``.
            num_phi_layers: Depth of the map learner.
            shared_phi: Share one map learner across layers.
            pe_size: Width of the positional encodings passed to ``forward``.
            num_node_types: Number of node types (used when
                ``use_node_types=True``).
            num_edge_types: Number of edge types (used when
                ``use_edge_types=True``).
            use_features: Feed the node features to the map learner.
            use_node_types: Feed node-type one-hots to the map learner.
            use_edge_types: Give the SumGNN map learner relation-specific
                neighbour weights.
            orth_strategy: Orthogonal parameterisation for ``bundle_dim != 2``.
            clamp_val: Clamp on the orthogonal-transform parameters.
            input_dropout: Dropout on raw input features before encoding.
            dropout: Dropout on hidden states between layers.
            normalize_output: L2-normalise the representation before decoding.
            jknet: Concatenate every layer's hidden state before the decoder.

        Raises:
            ValueError: On non-positive dimensions/layers, an unknown ``phi``,
                or an inconsistent het configuration.
        """
        super().__init__()
        if bundle_dim <= 0:
            raise ValueError("bundle_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_bundles <= 0:
            raise ValueError("num_bundles must be positive")
        if num_layers <= 0:
            raise ValueError("must have at least one HetBuNN layer")
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

        learner_kwargs: dict[str, Any] = dict(
            num_bundles=num_bundles,
            bundle_dim=bundle_dim,
            phi=phi,
            orth_strategy=orth_strategy,
            clamp_val=clamp_val,
            pe_size=pe_size,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            use_features=use_features,
            use_node_types=use_node_types,
            use_edge_types=use_edge_types,
        )
        self.shared_map_learner = (
            HetBundleMapLearner(
                context_dim=self.width,
                num_layers=num_phi_layers,
                **learner_kwargs,
            )
            if shared_phi
            else None
        )

        self.layers = nn.ModuleList(
            [
                HetBuNNConv(
                    in_channels=hidden_dim,
                    tau=tau,
                    learn_tau=learn_tau,
                    solver=solver,
                    max_degree=max_degree,
                    num_eigs=num_eigs,
                    bias=bias,
                    context_dim=self.width,
                    normalize=normalize,
                    num_phi_layers=num_phi_layers,
                    external_maps=shared_phi,
                    **learner_kwargs,
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

        # Per-graph eigendecomposition cache for the spectral solver, as in
        # BuNNModel: one entry, which serves transductive node tasks and never
        # hits under graph-level batching (prefer solver="taylor" there).
        self._eig_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        if self.shared_map_learner is not None:
            self.shared_map_learner.reset_parameters()
        for layer in self.layers:
            assert isinstance(layer, HetBuNNConv)
            layer.reset_parameters()
        # `decoder` may have been replaced with nn.Identity for graph-level
        # pooling, which has no parameters to reset.
        if isinstance(self.decoder, nn.Linear):
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
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Runs the HetBuNN encoder, diffusion layers, and decoder.

        Args:
            x: Raw node features ``[num_nodes, in_channels]``.
            edge_index: Graph connectivity ``[2, num_edges]``, assumed
                undirected (both directions stored).
            node_types: Node-type indices ``[num_nodes]`` or None.
            edge_types: Edge-type indices ``[num_edges]`` or None.
            pe: Positional encodings ``[num_nodes, pe_size]`` fed to
                :math:`\phi`. Required when the model was built with
                ``pe_size > 0``.

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
                node_rep = self.shared_map_learner(
                    h, edge_index, node_types, edge_types, pe
                )
            h = layer(
                h,
                h,
                edge_index,
                node_types,
                edge_types,
                node_rep,
                eig_vecs,
                eig_vals,
                pe,
            )
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


__all__ = ["HetBuNNModel"]
