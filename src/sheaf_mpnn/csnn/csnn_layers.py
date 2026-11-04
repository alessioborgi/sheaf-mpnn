# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Cooperative Sheaf Neural Network convolution and its conformal map learner."""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GraphSAGE, MessagePassing
from torch_geometric.utils import scatter

from sheaf_mpnn.csnn.gnn_builder import build_gnn, get_layer_type
from sheaf_mpnn.utils import build_orthogonal, orth_param_count

# Upper bound on the conformal scale, matching the reference clamp of log(10)
# before the exponential (so c lies in (0, 10]).
_LOG10 = math.log(10.0)


class ConformalSheafLearner(nn.Module):
    """Predicts per-node conformal restriction maps.

    Runs node features through a GNN (or MLP) and splits the output into
    orthogonal-map parameters and, for the conformal variant, a positive scale
    ``c``.  The restriction map attached to a node is then ``c * Q``.

    Unlike the map generators elsewhere in ``sheaf_mpnn``, which are per-edge
    linear layers, this learner is **per-node**: it emits one map per node,
    which the convolution lifts to edges inside ``message``.
    """

    def __init__(
        self,
        stalk_dim: int,
        hidden_dim: int,
        num_params: int,
        conformal: bool = True,
        gnn_layers: int = 0,
        gnn_hidden: int = 16,
        gnn_default: int = 1,
        gnn_type: str = "SAGE",
        gnn_residual: bool = False,
        linear_emb: bool = False,
        pe_size: int = 0,
    ):
        """Initializes the conformal sheaf learner.

        Args:
            stalk_dim: Stalk dimension ``d``.
            hidden_dim: Feature width per stalk entry.
            num_params: Number of free orthogonal-map parameters per node.
            conformal: If ``True``, emit an extra channel decoded as a positive
                scale ``c``; otherwise ``c == 1``.
            gnn_layers: Backbone depth. ``0`` selects a two-layer MLP.
            gnn_hidden: Hidden width of the backbone.
            gnn_default: ``1`` and ``2`` select the two reference ``GraphSAGE``
                presets; ``0`` builds a custom stack from ``gnn_type``.
            gnn_type: Convolution type used when ``gnn_default == 0``.
            gnn_residual: Adds residual connections in the custom stack.
            linear_emb: Wraps the custom stack in linear embed/project layers.
            pe_size: Width of the optional positional encoding concatenated to
                the node features.

        Raises:
            ValueError: If ``gnn_default`` is not 0, 1 or 2, if ``gnn_type``
                is ``SGC`` while ``gnn_residual`` is set, or if ``gnn_type`` is
                ``GPS`` on the custom path without ``linear_emb``.
        """
        super().__init__()
        if gnn_default not in (0, 1, 2):
            raise ValueError(f"gnn_default must be 0, 1 or 2, got {gnn_default!r}")
        if (gnn_type, gnn_residual) == ("SGC", True):
            raise ValueError("SGC does not support residual connections.")
        if gnn_default == 0 and gnn_layers > 0 and gnn_type == "GPS" and not linear_emb:
            raise ValueError(
                "gnn_type='GPS' requires linear_emb=True: GPSConv preserves its "
                "input width, so only the embed/project wrapper can produce the "
                "map-parameter width."
            )

        self.conformal = conformal
        self.gnn_layers = gnn_layers
        self.gnn_default = gnn_default
        self.gnn_type = gnn_type
        self.residual = gnn_residual
        self.linear_emb = linear_emb
        self.pe_size = pe_size

        # The learner input is the flattened stalk with the positional encoding
        # appended, so the encoding contributes pe_size columns in total. (The
        # reference sizes some branches as (hidden_dim + pe_size) * stalk_dim,
        # which only agrees with this for pe_size == 0 -- the case every shipped
        # config uses. We use the consistent width so pe_size > 0 works.)
        in_dim = stalk_dim * hidden_dim + pe_size
        out_dim = num_params + 1 if conformal else num_params
        self.out_dim = out_dim

        self.emb1: nn.Module | None = None
        self.emb2: nn.Module | None = None

        if gnn_layers > 0:
            self.conv_cls = get_layer_type(gnn_type)
            self._conv_accepts_edge_attr = (
                "edge_attr" in inspect.signature(self.conv_cls.forward).parameters
            )
            if gnn_default == 1:
                self.phi: nn.Module = GraphSAGE(
                    in_dim,
                    gnn_hidden,
                    num_layers=gnn_layers,
                    out_channels=out_dim,
                    norm="layer",
                )
            elif gnn_default == 2:
                self.phi = GraphSAGE(
                    in_dim,
                    gnn_hidden,
                    num_layers=gnn_layers,
                    out_channels=out_dim,
                    dropout=0.2,
                    act="gelu",
                    project=True,
                    bias=True,
                )
            elif linear_emb:
                self.emb1 = nn.Linear(in_dim, gnn_hidden)
                self.phi = build_gnn(gnn_type, gnn_hidden, gnn_hidden, gnn_layers)
                self.emb2 = nn.Linear(gnn_hidden, out_dim)
            else:
                self.phi = build_gnn(gnn_type, in_dim, out_dim, gnn_layers, gnn_hidden)
        else:
            self.phi = nn.Sequential(
                nn.Linear(in_dim, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, out_dim),
            )

    def reset_parameters(self) -> None:
        for module in (self.emb1, self.emb2):
            if module is not None:
                assert isinstance(module, nn.Linear)
                module.reset_parameters()
        for module in self.phi.modules():
            reset = getattr(module, "reset_parameters", None)
            if module is not self.phi and callable(reset):
                reset()
        reset_phi = getattr(self.phi, "reset_parameters", None)
        if callable(reset_phi):
            reset_phi()

    def _accepts_edge_attr(self) -> bool:
        """Whether the configured convolution takes an ``edge_attr`` argument."""
        return self._conv_accepts_edge_attr

    def _run_stack(
        self,
        maps: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        *,
        always_act: bool,
    ) -> Tensor:
        """Runs the custom convolution stack built by ``build_gnn``.

        Args:
            maps: Node features ``[num_nodes, in_dim]``.
            edge_index: Graph connectivity ``[2, num_edges]``.
            edge_attr: Optional edge features, forwarded only to convolutions
                whose ``forward`` accepts them.
            always_act: If ``True``, apply GELU after every layer (the
                ``linear_emb`` path). Otherwise skip it on a single-layer stack
                and on the final layer, matching the reference.

        Returns:
            Node features after the stack.

        Raises:
            ValueError: If ``gnn_type`` is ``NNConv`` and no ``edge_attr`` was
                provided -- ``NNConv`` computes its weights from edge features.
        """
        if self.gnn_type == "NNConv" and edge_attr is None:
            raise ValueError(
                "gnn_type='NNConv' requires edge_attr, and none was provided. "
                "The node-classification pipeline passes no edge features, so "
                "NNConv is only usable from callers that supply edge_attr."
            )
        if self.gnn_type == "SGC":
            if isinstance(self.phi, nn.ModuleList):
                for layer in self.phi:
                    maps = layer(maps, edge_index)
                    if layer is not self.phi[-1]:
                        maps = F.gelu(maps)
            else:
                maps = self.phi(maps, edge_index)
                if always_act:
                    maps = F.gelu(maps)
            return maps

        pass_edge_attr = edge_attr is not None and self._accepts_edge_attr()
        assert isinstance(self.phi, nn.ModuleList)
        last = self.gnn_layers - 1
        for i in range(self.gnn_layers):
            prev = maps
            if pass_edge_attr:
                maps = self.phi[i](maps, edge_index, edge_attr=edge_attr)
            else:
                maps = self.phi[i](maps, edge_index)

            if always_act:
                maps = F.gelu(maps)
                if self.residual:
                    maps = maps + prev
            else:
                if self.gnn_layers != 1 and i != last:
                    maps = F.gelu(maps)
                # The reference skips the residual on the first and last layers,
                # where the width changes and the shapes would not match.
                if self.residual and i not in (0, last):
                    maps = maps + prev
        return maps

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        pe: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predicts orthogonal-map parameters and conformal scales.

        Args:
            x: Flattened stalk features ``[num_nodes, stalk_dim * hidden_dim]``.
            edge_index: Graph connectivity ``[2, num_edges]``.
            edge_attr: Optional edge features ``[num_edges, *]``.
            pe: Optional positional encoding ``[num_nodes, pe_size]``.

        Returns:
            A tuple ``(params, c)`` where ``params`` is
            ``[num_nodes, num_params]`` bounded to ``(-1, 1)`` and ``c`` is
            ``[num_nodes]`` of positive scales (all ones when not conformal).
        """
        if pe is not None:
            x = torch.cat([x, pe], dim=-1)

        if self.gnn_layers == 0:
            maps = self.phi(x)
        elif self.gnn_default:
            maps = self.phi(x, edge_index)
        elif self.linear_emb:
            assert self.emb1 is not None and self.emb2 is not None
            maps = F.gelu(self.emb1(x))
            maps = self._run_stack(maps, edge_index, edge_attr, always_act=True)
            maps = self.emb2(maps)
        else:
            maps = self._run_stack(x, edge_index, edge_attr, always_act=False)

        if self.conformal:
            return torch.tanh(maps[:, :-1]), torch.exp(maps[:, -1].clamp_max(_LOG10))
        return torch.tanh(maps), maps.new_ones(maps.size(0))


class CSNNConv(MessagePassing):
    """Cooperative sheaf convolution (Cooperative Sheaf Neural Networks).

    Applies two normalised sheaf Laplacians in sequence -- ``L_out`` then
    ``L_in`` -- built from independent per-node conformal restriction maps
    ``T`` (outgoing) and ``S`` (incoming), then combines the result with a
    learnable per-stalk skip::

        x <- (1 + tanh(eps)) * x0 - L x

    ``eps`` is zero-initialised, so the skip coefficient starts at exactly 1.

    This layer subclasses ``MessagePassing`` directly rather than
    ``BaseSheafConv``: the base builds **per-edge** maps from
    ``cat([x_dst, x_src])``, allocates ``W1``/``W2`` as raw parameters, fixes
    ``flow`` to the default, and has no notion of a conformal scale channel --
    none of which fit the cooperative formulation.
    """

    def __init__(
        self,
        stalk_dim: int,
        hidden_dim: int,
        variant_conformal: bool = True,
        orth_strategy: str = "householder",
        clamp_val: float = 10.0,
        left_weights: bool = True,
        right_weights: bool = True,
        use_act: bool = True,
        dropout: float = 0.0,
        gnn_layers: int = 0,
        gnn_hidden: int = 16,
        gnn_default: int = 1,
        gnn_type: str = "SAGE",
        gnn_residual: bool = False,
        linear_emb: bool = False,
        pe_size: int = 0,
        bias: bool = False,
    ):
        """Initializes a cooperative sheaf convolution.

        Args:
            stalk_dim: Stalk dimension ``d``. Must be greater than 1.
            hidden_dim: Feature width per stalk entry.
            variant_conformal: Whether the restriction maps carry a learned
                positive scale.
            orth_strategy: Orthogonal parameterization, one of
                ``ORTH_STRATEGIES``.
            clamp_val: Clamping bound for the Cayley transform.
            left_weights: Learn a ``d x d`` stalk-mixing transform.
            right_weights: Learn a ``hidden_dim x hidden_dim`` channel transform.
            use_act: Apply GELU to the diffusion term.
            dropout: Dropout applied to the diffusion term.
            gnn_layers: Sheaf-learner backbone depth.
            gnn_hidden: Sheaf-learner hidden width.
            gnn_default: Sheaf-learner preset (0, 1 or 2).
            gnn_type: Convolution type when ``gnn_default == 0``.
            gnn_residual: Residual connections in the custom learner stack.
            linear_emb: Linear embed/project wrapper for the custom stack.
            pe_size: Width of the optional positional encoding.
            bias: Bias for the left/right transforms.

        Raises:
            ValueError: If ``stalk_dim <= 1`` or ``hidden_dim <= 0``.
        """
        # flow="target_to_source" binds i = edge_index[0] and j = edge_index[1]
        # and scatters into edge_index[0]; node_dim=0 is required because the
        # propagated tensors are 3-D [N, d, hidden_dim]. Both differ from the
        # PyG defaults and from every other layer in this package.
        super().__init__(aggr="add", flow="target_to_source", node_dim=0)
        if stalk_dim <= 1:
            raise ValueError("stalk_dim must be greater than 1")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.d = stalk_dim
        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.orth_strategy = orth_strategy
        self.clamp_val = clamp_val
        self.use_act = use_act
        self.dropout = dropout

        self.lin_right_weights: nn.Module = (
            nn.Linear(hidden_dim, hidden_dim, bias=bias)
            if right_weights
            else nn.Identity()
        )
        self.lin_left_weights: nn.Module = (
            nn.Linear(stalk_dim, stalk_dim, bias=bias)
            if left_weights
            else nn.Identity()
        )

        num_params = orth_param_count(orth_strategy, stalk_dim)

        def make_learner() -> ConformalSheafLearner:
            return ConformalSheafLearner(
                stalk_dim=stalk_dim,
                hidden_dim=hidden_dim,
                num_params=num_params,
                conformal=variant_conformal,
                gnn_layers=gnn_layers,
                gnn_hidden=gnn_hidden,
                gnn_default=gnn_default,
                gnn_type=gnn_type,
                gnn_residual=gnn_residual,
                linear_emb=linear_emb,
                pe_size=pe_size,
            )

        # The outgoing and incoming maps are learned independently: same
        # architecture, separate weights.
        #
        # Named *_maps_learner rather than *_map_generator on purpose: the
        # Lightning module routes parameters matching "map_generator" into a
        # separate sheaf_decay group, while the reference CSNN optimizer uses a
        # single flat Adam group.
        self.out_maps_learner = make_learner()
        self.in_maps_learner = make_learner()

        self.epsilons = nn.Parameter(torch.zeros((stalk_dim, 1)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.lin_right_weights, nn.Linear):
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        if isinstance(self.lin_left_weights, nn.Linear):
            nn.init.eye_(self.lin_left_weights.weight.data)
        self.out_maps_learner.reset_parameters()
        self.in_maps_learner.reset_parameters()
        nn.init.zeros_(self.epsilons)

    def _restriction_maps(
        self,
        edge_index: Tensor,
        t_params: Tensor,
        s_params: Tensor,
        c_t: Tensor,
        c_s: Tensor,
        num_nodes: int,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Builds degree-normalised incoming and outgoing restriction maps.

        The ``+1`` in the degree terms is the identity contribution of the
        cooperative Laplacian, so callers must not also add self-loops.
        """
        row, col = edge_index

        t_map = build_orthogonal(t_params, self.d, self.orth_strategy, self.clamp_val)
        s_map = build_orthogonal(s_params, self.d, self.orth_strategy, self.clamp_val)
        t_map = t_map * c_t[:, None, None]
        s_map = s_map * c_s[:, None, None]

        in_diag = scatter(c_s[col] ** 2, col, dim=0, dim_size=num_nodes, reduce="sum")
        out_diag = scatter(c_t[row] ** 2, row, dim=0, dim_size=num_nodes, reduce="sum")
        in_diag = in_diag[:, None]
        out_diag = out_diag[:, None]

        out_sqrt_inv = (out_diag + 1).pow(-0.5).view(-1, 1, 1)
        in_sqrt_inv = (in_diag + 1).pow(-0.5).view(-1, 1, 1)

        in_comps = (in_diag, t_map * in_sqrt_inv, s_map * in_sqrt_inv)
        out_comps = (out_diag, t_map * out_sqrt_inv, s_map * out_sqrt_inv)
        return in_comps, out_comps

    def _left_right_linear(self, x: Tensor, num_nodes: int) -> Tensor:
        """Applies the ``d x d`` stalk transform then the channel transform.

        Args:
            x: Stalk features ``[num_nodes * d, hidden_dim]``.
            num_nodes: Number of nodes in the (possibly batched) graph.

        Returns:
            Transformed features of the same shape.
        """
        x = x.t().reshape(-1, self.d)
        x = self.lin_left_weights(x)
        x = x.reshape(-1, num_nodes * self.d).t()
        return self.lin_right_weights(x)

    def forward(
        self,
        x_feat: Tensor,
        x_stalk: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        pe: Tensor | None = None,
    ) -> Tensor:
        """Runs one cooperative sheaf diffusion step.

        Args:
            x_feat: Flattened stalk features ``[num_nodes, d * hidden_dim]``.
                Must be the same tensor as ``x_stalk``, reshaped -- the caller
                applies dropout once so the learner input and the skip term
                share a mask.
            x_stalk: Stalk features ``[num_nodes, d, hidden_dim]``.
            edge_index: Graph connectivity ``[2, num_edges]``.
            edge_attr: Optional edge features, forwarded to the sheaf learners.
            pe: Optional positional encoding, forwarded to the sheaf learners.

        Returns:
            Updated flattened stalk features ``[num_nodes, d * hidden_dim]``.
        """
        num_nodes = x_feat.size(0)

        t_params, c_t = self.out_maps_learner(x_feat, edge_index, edge_attr, pe)
        s_params, c_s = self.in_maps_learner(x_feat, edge_index, edge_attr, pe)

        x0 = x_stalk.reshape(num_nodes * self.d, self.hidden_dim)

        in_comps, out_comps = self._restriction_maps(
            edge_index, t_params, s_params, c_t, c_s, num_nodes
        )
        d_in, t_in, s_in = in_comps
        d_out, t_out, s_out = out_comps

        c_t_norm = c_t[:, None] ** 2 * (d_out + 1).pow(-1)
        c_s_norm = c_s[:, None] ** 2 * (d_in + 1).pow(-1)

        x = self._left_right_linear(x0, num_nodes)
        x = x.reshape(num_nodes, self.d, self.hidden_dim)

        # L_out, then L_in. The second pass deliberately reuses the output of
        # the first with the incoming normalisation -- this asymmetry is the
        # cooperative update, not an oversight.
        x = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            x=c_t_norm[..., None] * x,
            y=s_out @ x,
            T=t_out.transpose(-2, -1),
        )
        x = self.propagate(  # ty: ignore[missing-argument]
            edge_index,  # ty: ignore[invalid-argument-type]
            x=c_s_norm[..., None] * x,
            y=s_in @ x,
            T=t_in.transpose(-2, -1),
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_act:
            x = F.gelu(x)

        # Broadcasting epsilons [d, 1] over [N, d, hidden_dim] computes the
        # same per-stalk coefficient as the reference's tile over [N * d, 1].
        out = (1 + torch.tanh(self.epsilons)) * x0.view(
            num_nodes, self.d, self.hidden_dim
        ) - x
        return out.reshape(num_nodes, -1)

    def message(  # ty: ignore[invalid-method-override]
        self, x_i: Tensor, y_j: Tensor, T_i: Tensor
    ) -> Tensor:
        """One Laplacian row: the diagonal self term minus the transported neighbour.

        The diagonal term rides inside the message and is therefore summed once
        per incident edge; the ``(D + 1)`` normalisation applied to ``x`` is what
        makes that cancel correctly.
        """
        return x_i - T_i @ y_j


__all__ = ["CSNNConv", "ConformalSheafLearner"]
