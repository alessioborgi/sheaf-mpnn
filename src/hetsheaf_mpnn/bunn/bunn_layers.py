# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Heterogeneous Bundle Neural Network (HetBuNN) convolution layer."""

from typing import Literal

import torch

from hetsheaf_mpnn.bunn.bundle import HetBundleMapLearner
from sheaf_mpnn.bunn.bundle import PhiKind
from sheaf_mpnn.bunn.bunn_layers import BuNNConv, BuNNSolver, Normalization


class HetBuNNConv(BuNNConv):
    r"""One heterogeneous BuNN layer: type-aware bundle maps, type-blind diffusion.

    Identical to :class:`~sheaf_mpnn.bunn.bunn_layers.BuNNConv` except for the
    map learner, a :class:`~hetsheaf_mpnn.bunn.bundle.HetBundleMapLearner`
    that reads node types (``use_node_types``) and, with ``phi="sum_gnn"``,
    edge types (``use_edge_types``).

    The heat kernel is still the plain graph heat kernel. A flat bundle fixes
    the transport across an edge to :math:`O_v^\top O_u` whatever the edge's
    type, and that factorisation is what reduces bundle diffusion to
    synchronise, graph heat kernel, desynchronise (Lemma 3.1). Relation-specific
    transports would make the bundle non-flat and lose it, so edge types act
    through the maps instead.
    """

    def __init__(
        self,
        num_bundles: int,
        bundle_dim: int,
        in_channels: int,
        tau: float = 1.0,
        learn_tau: bool = False,
        solver: BuNNSolver = "taylor",
        max_degree: int | None = None,
        num_eigs: int | None = None,
        context_dim: int | None = None,
        pe_size: int = 0,
        normalize: Normalization = "sym",
        bias: bool = True,
        phi: PhiKind = "mlp",
        num_phi_layers: int = 2,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        external_maps: bool = False,
        num_node_types: int = 0,
        num_edge_types: int = 0,
        use_features: bool = True,
        use_node_types: bool = False,
        use_edge_types: bool = False,
    ):
        """Initializes a heterogeneous BuNN convolution layer.

        Args:
            num_bundles: Number of parallel bundles ``b``.
            bundle_dim: Stalk dimension ``d`` of each bundle.
            in_channels: Vector-field channels ``f`` per bundle.
            tau: Diffusion time ``t``.
            learn_tau: Learn one diffusion time per bundle.
            solver: ``"taylor"`` or ``"spectral"`` heat-kernel evaluation.
            max_degree: Taylor truncation degree (``"taylor"`` only).
            num_eigs: Spectral truncation (``"spectral"`` only).
            context_dim: Width of the node context features ``x_feat``.
                Defaults to the state width.
            pe_size: Width of the positional encodings fed to ``phi``.
            normalize: ``"sym"`` or ``"rw"`` normalised Laplacian.
            bias: Whether the Eq. 2 encoder carries a bias.
            phi: Map-learner backbone; ``use_edge_types`` needs ``"sum_gnn"``.
            num_phi_layers: Depth of the map learner.
            orth_strategy: Orthogonal parameterisation for ``bundle_dim != 2``.
            clamp_val: Clamp on the orthogonal-transform parameters.
            external_maps: If ``True``, the layer owns no map learner and
                ``forward`` requires ``node_rep`` (the model's shared ``phi``).
            num_node_types: Number of node types.
            num_edge_types: Number of edge types.
            use_features: Feed the node context features to ``phi``.
            use_node_types: Feed node-type one-hots to ``phi``.
            use_edge_types: Give the SumGNN ``phi`` relation-specific
                neighbour weights.

        Raises:
            ValueError: On anything :class:`BuNNConv` or
                :class:`HetBundleMapLearner` rejects.
        """
        # The base would build a type-blind phi; ask it for none and build the
        # type-aware one below.
        super().__init__(
            num_bundles=num_bundles,
            bundle_dim=bundle_dim,
            in_channels=in_channels,
            tau=tau,
            learn_tau=learn_tau,
            solver=solver,
            max_degree=max_degree,
            num_eigs=num_eigs,
            context_dim=context_dim,
            pe_size=pe_size,
            normalize=normalize,
            bias=bias,
            phi=phi,
            num_phi_layers=num_phi_layers,
            orth_strategy=orth_strategy,
            clamp_val=clamp_val,
            external_maps=True,
        )
        self.map_learner = (
            None
            if external_maps
            else HetBundleMapLearner(
                context_dim=self.context_dim,
                num_bundles=num_bundles,
                bundle_dim=bundle_dim,
                phi=phi,
                num_layers=num_phi_layers,
                orth_strategy=orth_strategy,
                clamp_val=clamp_val,
                pe_size=pe_size,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                use_features=use_features,
                use_node_types=use_node_types,
                use_edge_types=use_edge_types,
            )
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x_feat: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        node_rep: torch.Tensor | None = None,
        eig_vecs: torch.Tensor | None = None,
        eig_vals: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Applies one het BuNN layer: type-aware maps, then the BuNN update.

        Args:
            x_feat: Node context features ``[N, context_dim]`` driving
                :math:`\phi`.
            x: Flat node states ``[N, num_bundles * bundle_dim * in_channels]``.
            edge_index: Graph connectivity ``[2, E]``, assumed undirected.
            node_types: Node-type indices ``[N]`` or None.
            edge_types: Edge-type indices ``[E]`` or None.
            node_rep: Optional precomputed bundle maps ``[N, b, d, d]``;
                required when the layer was built with ``external_maps=True``.
            eig_vecs: Optional Laplacian eigenvectors ``[N, K]`` (spectral only).
            eig_vals: Optional Laplacian eigenvalues ``[K]`` (spectral only).
            pe: Positional encodings ``[N, pe_size]``; required when
                ``pe_size > 0``.

        Returns:
            Updated flat node states, same shape as ``x``.
        """
        # Validate the state before phi runs, as the base layer does.
        self._to_bundle(x)
        if node_rep is None:
            learner = self.map_learner
            if learner is None:
                raise ValueError(
                    "layer was built with external_maps=True; pass node_rep explicitly"
                )
            assert isinstance(learner, HetBundleMapLearner)
            node_rep = learner(x_feat, edge_index, node_types, edge_types, pe)
        return super().forward(x_feat, x, edge_index, node_rep, eig_vecs, eig_vals, pe)


__all__ = ["HetBuNNConv"]
