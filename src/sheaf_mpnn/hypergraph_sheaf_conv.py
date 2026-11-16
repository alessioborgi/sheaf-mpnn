# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Incidence-wise transport for real, undirected hypergraph sheaves."""

import math
from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from sheaf_mpnn.utils.normalization import batched_sym_matrix_pow_svd


class BaseHypergraphSheafConv(MessagePassing):
    r"""Shared two-stage HSNN propagation, without assembling a Laplacian.

    For incidence maps ``B[e, v] = F_ve``, this layer applies
    ``P = R B.T D_E^{-1} B R``, where ``D_v = sum_e F_ve.T F_ve`` and
    ``D_E[e] = |e| I``. By default ``R`` is the pseudoinverse square root of
    ``D``. Positive ``degree_regularization`` instead uses ``(D + tau I)^-1/2``.
    Zero eigenvalues are discarded using a relative, dtype-dependent tolerance.

    The update is ``ELU(P (W1 dropout(X) W2) + residual * X)``. This is the
    propagation form of HSNN, not the graph NSD residual Laplacian update.
    Each incidence has one map; the return pass uses that map's transpose.

    Args:
        stalk_dim: Positive stalk dimension d.
        in_channels: Positive number of feature channels f per stalk entry.
        hidden_dim: Positive compatibility argument for the NSD layer API.
            Does not set a predictor width: the learner is one linear layer.
        context_dim: Node context width C; defaults to d*f.
        dropout: Dropout probability on the diffusion input only.
        residual: Add the un-dropped input before the output activation.
        degree_regularization: Nonnegative tau; zero selects the pseudoinverse.
    """

    diagonal_maps: bool = False
    map_generator: nn.Sequential

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
        degree_regularization: float = 0.0,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if min(stalk_dim, in_channels, hidden_dim) <= 0:
            raise ValueError("stalk_dim, in_channels and hidden_dim must be positive")
        self.context_dim = (
            stalk_dim * in_channels if context_dim is None else context_dim
        )
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not math.isfinite(degree_regularization) or degree_regularization < 0:
            raise ValueError("degree_regularization must be finite and nonnegative")
        self.stalk_dim = stalk_dim
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.degree_regularization = degree_regularization
        self.W1 = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.W2 = nn.Parameter(torch.empty(in_channels, in_channels))
        self.sigma: nn.Module = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        """Reset the bilateral transforms and all map learner linear layers."""
        nn.init.eye_(self.W1)
        nn.init.orthogonal_(self.W2)
        if hasattr(self, "map_generator"):
            for module in self.map_generator:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def _build_map_generator(self, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * self.context_dim, out_dim, bias=False), nn.Tanh()
        )

    @staticmethod
    def _validate_incidence(
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_hyperedges: int | None,
        device: torch.device,
    ) -> int:
        if hyperedge_index.ndim != 2 or hyperedge_index.size(0) != 2:
            raise ValueError("hyperedge_index must have shape [2, num_incidences]")
        if hyperedge_index.dtype != torch.long:
            raise ValueError("hyperedge_index must have dtype torch.long")
        if hyperedge_index.device != device:
            raise ValueError("hyperedge_index and features must share a device")
        if num_hyperedges is not None and (
            not isinstance(num_hyperedges, int)
            or isinstance(num_hyperedges, bool)
            or num_hyperedges < 0
        ):
            raise ValueError("num_hyperedges must be a nonnegative integer")
        if hyperedge_index.numel() == 0:
            return 0 if num_hyperedges is None else num_hyperedges
        if bool((hyperedge_index < 0).any()):
            raise ValueError("hyperedge_index contains negative indices")
        if int(hyperedge_index[0].max()) >= num_nodes:
            raise ValueError("node index is out of range")
        required = int(hyperedge_index[1].max()) + 1
        if num_hyperedges is None:
            num_hyperedges = required
        elif required > num_hyperedges:
            raise ValueError("hyperedge index is out of range")
        if hyperedge_index.unique(dim=1).size(1) != hyperedge_index.size(1):
            raise ValueError("duplicate node-hyperedge incidences are not allowed")
        return num_hyperedges

    def _validate_stalk(self, x_stalk: torch.Tensor) -> None:
        if (
            x_stalk.ndim != 3
            or x_stalk.size(0) == 0
            or x_stalk.shape[1:] != (self.stalk_dim, self.in_channels)
        ):
            raise ValueError("x_stalk must have shape [N > 0, stalk_dim, in_channels]")
        if x_stalk.dtype not in (torch.float32, torch.float64):
            raise ValueError("stalk features must have dtype float32 or float64")

    def get_restriction_maps(
        self,
        x_feat: torch.Tensor,
        hyperedge_index: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
    ) -> torch.Tensor:
        """Predict one restriction map for each node-hyperedge incidence.

        Args:
            x_feat: Node contexts [N, C], matching the learner's device/dtype.
            hyperedge_index: Long indices [2, K], node row followed by edge row.
            num_hyperedges: Optional M, including trailing empty hyperedges.

        Returns:
            Maps [K, d] for diagonal layers, otherwise [K, d, d]. The learner
            sees each node context concatenated with its hyperedge's mean context.
        """
        self._validate_context(x_feat)
        num_hyperedges = self._validate_incidence(
            hyperedge_index, x_feat.size(0), num_hyperedges, x_feat.device
        )
        return self._predict_maps(x_feat, hyperedge_index, num_hyperedges)

    def _validate_context(self, x_feat: torch.Tensor) -> None:
        if (
            x_feat.ndim != 2
            or x_feat.size(0) == 0
            or x_feat.size(1) != self.context_dim
        ):
            raise ValueError("x_feat must have shape [N > 0, context_dim]")
        if x_feat.dtype != self.W1.dtype or x_feat.device != self.W1.device:
            raise ValueError("x_feat must match the layer's device and dtype")

    def _predict_maps(
        self,
        x_feat: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_hyperedges: int,
    ) -> torch.Tensor:
        """Predict maps from already validated contexts and incidences."""
        node, edge = hyperedge_index
        edge_context = scatter(
            x_feat[node], edge, dim=0, dim_size=num_hyperedges, reduce="mean"
        )
        context = torch.cat((x_feat[node], edge_context[edge]), dim=-1)
        return self.parameterize_maps(context)

    @abstractmethod
    def parameterize_maps(self, context: torch.Tensor) -> torch.Tensor:
        """Convert concatenated incidence contexts [K, 2*C] to restriction maps."""
        raise NotImplementedError

    def _normalization(
        self, maps: torch.Tensor, node: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        if self.diagonal_maps:
            degree = scatter(
                maps.square(), node, dim=0, dim_size=num_nodes, reduce="sum"
            )
            degree = degree + self.degree_regularization
            # Match the full-matrix helper's relative pseudoinverse threshold.
            tolerance = (
                degree.amax(dim=-1, keepdim=True)
                * self.stalk_dim
                * torch.finfo(degree.dtype).eps
            )
            good = degree > tolerance
            safe = torch.where(good, degree, torch.ones_like(degree))
            return torch.where(good, safe.rsqrt(), torch.zeros_like(degree))
        products = maps.transpose(-1, -2) @ maps
        degree = scatter(products, node, dim=0, dim_size=num_nodes, reduce="sum")
        identity = torch.eye(self.stalk_dim, dtype=maps.dtype, device=maps.device)
        degree = degree + self.degree_regularization * identity
        return batched_sym_matrix_pow_svd(degree, -0.5)

    def _multiply(self, maps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return maps.unsqueeze(-1) * x if self.diagonal_maps else maps @ x

    def apply_diffusion(
        self,
        x_stalk: torch.Tensor,
        hyperedge_index: torch.Tensor,
        restriction_maps: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
    ) -> torch.Tensor:
        """Apply only P to supplied stalks, with externally supplied maps.

        Args:
            x_stalk: Real features [N, d, f], float32 or float64.
            hyperedge_index: Long incidence indices [2, K]. Duplicates are invalid.
            restriction_maps: [K, d] or [K, d, d], according to the layer family.
                Must share the features' device/dtype and contain finite entries.
            num_hyperedges: Optional M; zero-incidence inputs are supported.

        Returns:
            Normalized propagation [N, d, f], without transforms, activation or
            residual. Isolated nodes and unsupported degree directions give zero.
        """
        self._validate_stalk(x_stalk)
        num_nodes = x_stalk.size(0)
        num_hyperedges = self._validate_incidence(
            hyperedge_index, num_nodes, num_hyperedges, x_stalk.device
        )
        return self._apply_diffusion(
            x_stalk, hyperedge_index, restriction_maps, num_hyperedges
        )

    def _apply_diffusion(
        self,
        x_stalk: torch.Tensor,
        hyperedge_index: torch.Tensor,
        restriction_maps: torch.Tensor,
        num_hyperedges: int,
    ) -> torch.Tensor:
        """Check maps and propagate with validated stalks and incidences.

        Shared by both public entry points so forward validates topology only
        once. No topology or learned tensors are cached between calls.
        """
        num_nodes = x_stalk.size(0)
        expected: tuple[int, ...] = (hyperedge_index.size(1), self.stalk_dim)
        if not self.diagonal_maps:
            expected = (*expected, self.stalk_dim)
        if restriction_maps.shape != expected:
            raise ValueError(f"restriction_maps must have shape {expected}")
        if (
            restriction_maps.dtype != x_stalk.dtype
            or restriction_maps.device != x_stalk.device
        ):
            raise ValueError("restriction_maps must match the features' device/dtype")
        if not bool(torch.isfinite(restriction_maps).all()):
            raise ValueError("restriction_maps must contain finite values")
        node, edge = hyperedge_index
        norm = self._normalization(restriction_maps, node, num_nodes)
        normalized = self._multiply(norm, x_stalk)
        edge_state = self.propagate(  # ty: ignore[missing-argument]
            hyperedge_index,  # ty: ignore[invalid-argument-type]
            x=(normalized, None),
            restriction_maps=restriction_maps,
            size=(num_nodes, num_hyperedges),
        )
        cardinality = scatter(
            x_stalk.new_ones(edge.numel()),
            edge,
            dim=0,
            dim_size=num_hyperedges,
            reduce="sum",
        )
        edge_state = edge_state / cardinality.clamp_min(1).view(-1, 1, 1)
        return_maps = (
            restriction_maps
            if self.diagonal_maps
            else restriction_maps.transpose(-1, -2)
        )
        returned = self.propagate(  # ty: ignore[missing-argument]
            hyperedge_index.flip(0),  # ty: ignore[invalid-argument-type]
            x=(edge_state, None),
            restriction_maps=return_maps,
            size=(num_hyperedges, num_nodes),
        )
        return self._multiply(norm, returned)

    def message(  # ty: ignore[invalid-method-override]
        self, x_j: torch.Tensor, restriction_maps: torch.Tensor
    ) -> torch.Tensor:
        """Transport source stalks through per-incidence maps [K, d(, d)]."""
        return self._multiply(restriction_maps, x_j)

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        hyperedge_index: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
    ) -> torch.Tensor:
        """Apply transforms, two-stage propagation, optional residual and ELU.

        Args:
            x_feat: Node contexts [N, C] on the layer's device and dtype.
            x_stalk: Node stalks [N, d, f] on the same device and dtype.
            hyperedge_index: Long incidence indices [2, K], node row then edge row.
            num_hyperedges: Optional M, including empty hyperedges.

        Returns:
            Updated node stalks [N, d, f].
        """
        self._validate_stalk(x_stalk)
        self._validate_context(x_feat)
        if x_feat.size(0) != x_stalk.size(0):
            raise ValueError("x_feat and x_stalk must have the same node count")
        if x_stalk.dtype != self.W1.dtype or x_stalk.device != self.W1.device:
            raise ValueError("x_stalk must match the layer's device and dtype")
        num_hyperedges = self._validate_incidence(
            hyperedge_index, x_stalk.size(0), num_hyperedges, x_stalk.device
        )
        maps = self._predict_maps(x_feat, hyperedge_index, num_hyperedges)
        z = self.W1 @ self.dropout_layer(x_stalk) @ self.W2
        out = self._apply_diffusion(z, hyperedge_index, maps, num_hyperedges)
        if self.residual:
            out = out + x_stalk
        return self.sigma(out)


__all__ = ["BaseHypergraphSheafConv"]
