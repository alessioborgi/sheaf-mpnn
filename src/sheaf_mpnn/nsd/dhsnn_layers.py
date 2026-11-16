"""Real restriction-map families with magnetic directed message passing.

Standalone convolutions reuse HSNN parameterizations. The complete directed
model uses the stalk-mean DirectedMapLearner below. Reference attribution and
Apache-2.0 terms are in third_party/dshn/NOTICE and LICENSE.
"""

import torch
from torch import nn
from torch_geometric.utils import scatter

from sheaf_mpnn.directed_hypergraph_sheaf_conv import BaseDirectedHypergraphSheafConv
from sheaf_mpnn.nsd.hsnn_layers import (
    DiagonalHSNNConv,
    GeneralHSNNConv,
    LowRankHSNNConv,
    OrthogonalHSNNConv,
)


class DirectedMapLearner(nn.Module):
    """Stalk-mean incidence predictor in the official DSHN layout.

    Returned full maps are transposed to the library's edge-by-node convention.
    Low rank and FastH strategies are library extensions to the reference.
    """

    def __init__(
        self,
        width,
        d,
        family,
        activation="sigmoid",
        prediction="MLP_var1",
        orth_strategy="cayley",
        rank=1,
        dropout=0.0,
        special_head=False,
    ):
        super().__init__()
        from sheaf_mpnn.utils.orthogonal import orth_param_count

        if prediction not in ("MLP_var1", "MLP_var2", "MLP_var3"):
            raise ValueError("unknown directed map prediction")
        if activation not in ("sigmoid", "tanh", "relu", "none"):
            raise ValueError("unknown map activation")
        self.d, self.family, self.activation, self.prediction = (
            d,
            family,
            activation,
            prediction,
        )
        self.orth_strategy, self.rank = orth_strategy, rank
        self.special_head = special_head
        outputs = (
            d
            if family == "diagonal"
            else 2 * d * rank
            if family == "low_rank"
            else d * d
        )
        if family == "orthogonal":
            outputs = (
                d * (d + 1) // 2
                if orth_strategy in ("cayley", "matrix_exp")
                else d * (d - 1) // 2
                if orth_strategy == "euler"
                else orth_param_count(orth_strategy, d)
            )
            if orth_strategy == "euler" and d not in (2, 3):
                raise ValueError("Euler maps require stalk_dim 2 or 3")
        self.linear = nn.Linear(4 * width, outputs, bias=False)
        self.edge_linear = (
            nn.Linear(width, width, bias=False) if prediction == "MLP_var3" else None
        )
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.edge_linear is not None:
            self.edge_linear.reset_parameters()

    def forward(self, x, edge_context, index, num_edges):
        node, edge = index
        context = x.mean(1)
        if self.prediction == "MLP_var1":
            edge_values = edge_context.mean(1)
        else:
            values = context
            if self.edge_linear is not None:
                values = torch.complex(
                    self.edge_linear(values.real), self.edge_linear(values.imag)
                )
            reduce = "mean" if self.prediction == "MLP_var2" else "sum"
            edge_values = torch.complex(
                scatter(values.real[node], edge, dim_size=num_edges, reduce=reduce),
                scatter(values.imag[node], edge, dim_size=num_edges, reduce=reduce),
            )
        features = torch.cat(
            (
                context[node].real,
                context[node].imag,
                edge_values[edge].real,
                edge_values[edge].imag,
            ),
            -1,
        )
        raw = self.linear(features)
        if self.activation != "none":
            raw = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}[
                self.activation
            ](raw)
        if self.family == "diagonal":
            maps = self.dropout(raw)
            if self.special_head:
                maps = torch.cat((maps[:, :-1], torch.ones_like(maps[:, -1:])), -1)
            return maps
        if self.family == "orthogonal":
            from sheaf_mpnn.utils.orthogonal import build_orthogonal

            if self.orth_strategy in ("cayley", "matrix_exp"):
                a = raw.new_zeros(raw.size(0), self.d, self.d)
                rows, cols = torch.tril_indices(self.d, self.d, device=raw.device)
                a[:, rows, cols] = raw
                a = a - a.mT
                eye = torch.eye(self.d, device=raw.device, dtype=raw.dtype)
                maps = (
                    torch.matrix_exp(a)
                    if self.orth_strategy == "matrix_exp"
                    else torch.linalg.solve(eye - a / 2, eye + a / 2)
                )
            elif self.orth_strategy == "euler":
                angles = raw * (2 * torch.pi)
                if self.d == 2:
                    c, s = angles.cos(), angles.sin()
                    maps = torch.cat((c, -s, s, c), -1).reshape(-1, 2, 2)
                else:
                    a, b, g = angles.unbind(-1)
                    ca, cb, cg = a.cos(), b.cos(), g.cos()
                    sa, sb, sg = a.sin(), b.sin(), g.sin()
                    maps = torch.stack(
                        (
                            ca * cb,
                            ca * sb * sg - sa * cg,
                            ca * sb * cg + sa * sg,
                            sa * cb,
                            sa * sb * sg + ca * cg,
                            sa * sb * cg - ca * sg,
                            -sb,
                            cb * sg,
                            cb * cg,
                        ),
                        -1,
                    ).reshape(-1, 3, 3)
            else:
                # The native compact product is equivalent to orgqr and avoids
                # loading an optional compiled extension in the directed model.
                strategy = (
                    "fasthpp"
                    if self.orth_strategy == "householder"
                    else self.orth_strategy
                )
                maps = build_orthogonal(raw, self.d, strategy)
        elif self.family == "low_rank":
            a, b = raw.reshape(-1, self.d, 2 * self.rank).chunk(2, -1)
            maps = a @ b.mT
        else:
            maps = raw.reshape(-1, self.d, self.d)
        maps = self.dropout(maps)
        if self.special_head and self.family == "orthogonal":
            mask = torch.ones(self.d, device=maps.device, dtype=maps.dtype)
            mask[-1] = 0
            maps = maps * mask[:, None] * mask[None, :]
            maps = maps + torch.diag(1 - mask)
        return maps.mT


class DiagonalDHSNNConv(BaseDirectedHypergraphSheafConv, DiagonalHSNNConv):
    """Directed hypergraph sheaf convolution with diagonal real maps."""


class GeneralDHSNNConv(BaseDirectedHypergraphSheafConv, GeneralHSNNConv):
    """Directed hypergraph sheaf convolution with full real maps."""


class OrthogonalDHSNNConv(BaseDirectedHypergraphSheafConv, OrthogonalHSNNConv):
    """Directed hypergraph sheaf convolution with orthogonal real maps."""


class LowRankDHSNNConv(BaseDirectedHypergraphSheafConv, LowRankHSNNConv):
    """Directed hypergraph sheaf convolution with pure low-rank real maps."""


__all__ = [
    "DiagonalDHSNNConv",
    "GeneralDHSNNConv",
    "OrthogonalDHSNNConv",
    "LowRankDHSNNConv",
]
