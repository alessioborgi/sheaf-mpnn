"""Magnetic two-stage message passing for directed hypergraph sheaves.

B[e,v] = exp(-2*pi*i*q) F_ve on tails and F_ve on heads. The return
transport is the Hermitian adjoint; learned restriction maps remain real.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from sheaf_mpnn.hypergraph_sheaf_conv import BaseHypergraphSheafConv


def complex_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop complex entries with one shared real mask for both components."""
    return x * F.dropout(torch.ones_like(x.real), p=p, training=training)


def complex_activation(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the paper's real-part-gated ReLU or componentwise ELU."""
    if mode == "complex_relu":
        return x * (x.real >= 0)
    return torch.complex(F.elu(x.real), F.elu(x.imag))


class BaseDirectedHypergraphSheafConv(BaseHypergraphSheafConv):
    r"""Apply P = R B^H D_E^-1 B R while retaining complex stalk states.

    Real map parameterizations and degree normalization are inherited from
    HSNN. ``incidence_roles`` is long[K], tail=-1, head=+1. Single-sided edges
    are allowed: their common phase cancels. Contexts are real unwindings of
    complex states and do not receive role labels directly.

    At singular degrees RLR = RDR-P, not I-P. Positive degree regularization
    also changes RDR. This class deliberately propagates with P in both cases.
    """

    def __init__(
        self,
        *args,
        q: float = 0.25,
        activation: Literal["complex_relu", "componentwise_elu"] = "complex_relu",
        norm_type: str | None = None,
        add_identity: bool = True,
        left_projection: bool = True,
        input_norm: bool = False,
        affine_norm: bool = True,
        light: bool = False,
        **kwargs,
    ) -> None:
        if not math.isfinite(q):
            raise ValueError("q must be finite")
        if activation not in ("complex_relu", "componentwise_elu"):
            raise ValueError("unknown complex activation")
        super().__init__(*args, **kwargs)
        self.q = float(q)
        self.activation = activation
        if norm_type not in (
            None,
            "degree_norm",
            "sym_degree_norm",
            "block_norm",
            "sym_block_norm",
        ):
            raise ValueError("unknown directed normalization")
        self.norm_type = norm_type
        self.light = light
        self.add_identity = add_identity
        self.left_projection = left_projection
        # None denotes the mathematical fixed-map operator used by the public
        # convolution API. Model layers always select an explicit normalization.
        self.feature_norm = (
            ComplexFeatureNorm(self.in_channels, affine_norm)
            if input_norm
            else torch.nn.Identity()
        )
        self.stalk_norm = (
            ComplexFeatureNorm(self.stalk_dim, affine_norm)
            if input_norm and left_projection
            else torch.nn.Identity()
        )
        if not left_projection:
            identity = torch.eye(self.stalk_dim)
            del self.W1
            self.register_buffer("W1", identity)
        self.bias = (
            torch.nn.Parameter(torch.zeros(self.in_channels))
            if norm_type is not None
            else None
        )
        if norm_type is not None:
            torch.nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
            if left_projection:
                torch.nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))

    def reset_parameters(self):
        """Reset transforms, predictor and reference normalization parameters."""
        super().reset_parameters()
        if getattr(self, "norm_type", None) is not None:
            torch.nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
            if self.left_projection:
                torch.nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
            assert self.bias is not None
            torch.nn.init.zeros_(self.bias)
            for norm in (self.feature_norm, self.stalk_norm):
                if isinstance(norm, ComplexFeatureNorm):
                    norm.reset_parameters()

    def _directed_norm(self, maps, node, num_nodes):
        if self.norm_type is None:
            return self._normalization(maps, node, num_nodes)
        symmetric = self.norm_type.startswith("sym_")
        power = -0.5 if symmetric else -1.0
        shift = float(self.add_identity) + self.degree_regularization
        if "degree" in self.norm_type:
            multiplicity = 1 if self.diagonal_maps else self.stalk_dim
            degree = (
                scatter(
                    maps.new_full((node.numel(),), multiplicity),
                    node,
                    dim_size=num_nodes,
                    reduce="sum",
                )
                + shift
            )
            values = degree.pow(power)
            values = torch.where(torch.isfinite(values), values, 0)
            return (
                values[:, None].expand(-1, self.stalk_dim)
                if self.diagonal_maps
                else torch.diag_embed(values[:, None].expand(-1, self.stalk_dim))
            )
        if self.diagonal_maps:
            degree = (
                scatter(maps.square(), node, dim_size=num_nodes, reduce="sum") + shift
            )
            return degree.clamp_min(1e-6).pow(power)
        # H blocks are F.T. The reference general block normalization always
        # adds I and detaches its inverse power, even when add_identity=False.
        from sheaf_mpnn.utils.normalization import batched_sym_matrix_pow_svd

        degree = scatter(maps.mT @ maps, node, dim_size=num_nodes, reduce="sum")
        eye = torch.eye(self.stalk_dim, device=maps.device, dtype=maps.dtype)
        return batched_sym_matrix_pow_svd(
            degree + (1 + self.degree_regularization) * eye, power
        ).detach()

    def _validate_stalk(self, x_stalk: torch.Tensor) -> None:
        if (
            x_stalk.ndim != 3
            or x_stalk.size(0) == 0
            or x_stalk.shape[1:] != (self.stalk_dim, self.in_channels)
        ):
            raise ValueError("x_stalk must have shape [N > 0, stalk_dim, in_channels]")
        if x_stalk.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("x_stalk must have dtype complex64 or complex128")

    @staticmethod
    def _validate_roles(roles: torch.Tensor, index: torch.Tensor) -> None:
        if roles.shape != (index.size(1),) or roles.dtype != torch.long:
            raise ValueError("incidence_roles must be long[num_incidences]")
        if roles.device != index.device:
            raise ValueError("incidence_roles and incidence index must share a device")
        if not bool(((roles == -1) | (roles == 1)).all()):
            raise ValueError("incidence_roles must contain only tail=-1 or head=+1")

    def _multiply(self, maps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        maps = maps.to(x.dtype)
        return maps.unsqueeze(-1) * x if self.diagonal_maps else maps @ x

    def _directed_diffusion(self, x_stalk, index, maps, roles, num_edges):
        num_nodes = x_stalk.size(0)
        expected = (index.size(1), self.stalk_dim)
        if not self.diagonal_maps:
            expected = (*expected, self.stalk_dim)
        if maps.shape != expected:
            raise ValueError(f"restriction_maps must have shape {expected}")
        if maps.dtype != x_stalk.real.dtype or maps.device != x_stalk.device:
            raise ValueError(
                "restriction_maps must match the real features device/dtype"
            )
        if not bool(torch.isfinite(maps).all()):
            raise ValueError("restriction_maps must contain finite values")
        node, edge = index
        # Official Light sparse products detach the assembled operator. Detach
        # its factors here, retaining gradients through transported features.
        if self.light:
            maps = maps.detach()
        norm = self._directed_norm(maps, node, num_nodes)
        # Reduce modulo one to avoid unnecessarily large trigonometric arguments.
        angle = -2 * math.pi * (self.q % 1)
        real, imag = math.cos(angle), math.sin(angle)
        real = 0.0 if abs(real) < 1e-15 else real
        imag = 0.0 if abs(imag) < 1e-15 else imag
        phase = torch.where(
            roles == -1,
            torch.tensor(
                complex(real, imag), device=x_stalk.device, dtype=x_stalk.dtype
            ),
            torch.tensor(1, device=x_stalk.device, dtype=x_stalk.dtype),
        )
        shape = (-1, 1) if self.diagonal_maps else (-1, 1, 1)
        transports = maps.to(x_stalk.dtype) * phase.view(shape)
        normalized = (
            self._multiply(norm, x_stalk)
            if self.norm_type is None or self.norm_type.startswith("sym_")
            else x_stalk
        )
        edge_state = self.propagate(  # ty: ignore[missing-argument]
            index,
            x=(normalized, None),
            restriction_maps=transports,
            size=(num_nodes, num_edges),
        )
        counts = scatter(
            maps.new_ones(edge.numel()), edge, dim=0, dim_size=num_edges, reduce="sum"
        )
        factor = (
            self.stalk_dim
            if self.norm_type is not None and not self.diagonal_maps
            else 1
        )
        edge_state = edge_state / (counts.clamp_min(1) * factor).view(-1, 1, 1)
        adjoints = transports.conj()
        if not self.diagonal_maps:
            adjoints = adjoints.transpose(-1, -2)
        returned = self.propagate(  # ty: ignore[missing-argument]
            index.flip(0),
            x=(edge_state, None),
            restriction_maps=adjoints,
            size=(num_edges, num_nodes),
        )
        return self._multiply(norm, returned)

    def apply_diffusion(  # ty: ignore[invalid-method-override]
        self,
        x_stalk: torch.Tensor,
        hyperedge_index: torch.Tensor,
        restriction_maps: torch.Tensor,
        incidence_roles: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
    ) -> torch.Tensor:
        """Propagate supplied complex states with supplied real restriction maps."""
        self._validate_stalk(x_stalk)
        num_edges = self._validate_incidence(
            hyperedge_index, x_stalk.size(0), num_hyperedges, x_stalk.device
        )
        self._validate_roles(incidence_roles, hyperedge_index)
        return self._directed_diffusion(
            x_stalk, hyperedge_index, restriction_maps, incidence_roles, num_edges
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        hyperedge_index: torch.Tensor,
        incidence_roles: torch.Tensor,
        *,
        num_hyperedges: int | None = None,
        activate: bool = True,
        restriction_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Learn maps, transform complex features, propagate and activate."""
        self._validate_stalk(x_stalk)
        self._validate_context(x_feat)
        if x_feat.size(0) != x_stalk.size(0):
            raise ValueError("contexts and stalks must have the same node count")
        if x_stalk.real.dtype != self.W1.dtype or x_stalk.device != self.W1.device:
            raise ValueError("x_stalk must match the layer's real device/dtype")
        num_edges = self._validate_incidence(
            hyperedge_index, x_stalk.size(0), num_hyperedges, x_stalk.device
        )
        self._validate_roles(incidence_roles, hyperedge_index)
        maps = (
            self._predict_maps(x_feat, hyperedge_index, num_edges)
            if restriction_maps is None
            else restriction_maps
        )
        dropped = complex_dropout(x_stalk, self.dropout_layer.p, self.training)
        if self.left_projection:
            dropped = self.stalk_norm(
                (self.W1.to(x_stalk.dtype) @ dropped).transpose(-1, -2)
            ).transpose(-1, -2)
        z = self.feature_norm(dropped @ self.W2.to(x_stalk.dtype))
        out = self._directed_diffusion(
            z, hyperedge_index, maps, incidence_roles, num_edges
        )
        if self.residual:
            out = out + x_stalk
        if self.bias is not None:
            out = out + torch.complex(self.bias, self.bias)
        return complex_activation(out, self.activation) if activate else out


class ComplexFeatureNorm(torch.nn.Module):
    """Reference batch whitening of real/imaginary components per channel."""

    def __init__(self, width: int, affine: bool = True):
        super().__init__()
        self.weight = (
            torch.nn.Parameter(
                torch.eye(2)[:, :, None].repeat(1, 1, width) * 0.70710678
            )
            if affine
            else None
        )
        self.bias = torch.nn.Parameter(torch.zeros(2, width)) if affine else None

    def reset_parameters(self):
        if self.weight is not None:
            assert self.bias is not None
            with torch.no_grad():
                self.weight.copy_(
                    torch.eye(2, device=self.weight.device, dtype=self.weight.dtype)[
                        :, :, None
                    ]
                    * 0.70710678
                )
                self.bias.zero_()

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        real = flat.real - flat.real.mean(0)
        imag = flat.imag - flat.imag.mean(0)
        rr, ii, ri = real.square().mean(0), imag.square().mean(0), (real * imag).mean(0)
        # A covariance determinant is nonnegative. Nearly collinear components
        # can yield a negative float32 value through cancellation, causing NaNs.
        s = ((rr * ii - ri.square()).clamp_min(0) + 1e-5).sqrt()
        t = (rr + ii + 2 * s + 1e-5).sqrt()
        inv = 1 / (s * t + 1e-5)
        a, b = ((ii + s) * real - ri * imag) * inv, ((rr + s) * imag - ri * real) * inv
        if self.weight is not None:
            assert self.bias is not None
            a, b = (
                self.weight[0, 0] * a + self.weight[0, 1] * b + self.bias[0],
                self.weight[1, 0] * a + self.weight[1, 1] * b + self.bias[1],
            )
        return torch.complex(a, b).reshape_as(x)


__all__ = ["BaseDirectedHypergraphSheafConv"]
