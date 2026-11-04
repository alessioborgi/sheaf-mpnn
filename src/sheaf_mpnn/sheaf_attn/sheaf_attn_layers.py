# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
from torch import nn
from torch_geometric.utils import add_self_loops, softmax

from sheaf_mpnn.base_conv import BaseSheafConv
from sheaf_mpnn.utils import build_orthogonal, orth_param_count


class SheafAttnConv(BaseSheafConv):
    """Sheaf Attention Network convolution layer.

    Extends BaseSheafConv with a GAT-style attention mechanism, replacing the
    degree-normalised sheaf Laplacian with an attention-weighted sheaf adjacency
    (Barbero et al., NeurIPS Workshop 2022).

    Implements both SheafAN (residual=False, Eq. 5) and Res-SheafAN
    (residual=True, Eq. 6).

    Restriction maps are orthogonal (O(stalk_dim)) parameterised via Cayley
    transform or Householder (fasth), with one independent map generator and
    attention vector per head; heads are averaged before propagation.  W1, W2,
    and reset_parameters are inherited from BaseSheafConv.
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        num_heads: int = 1,
        leaky_relu_slope: float = 0.2,
        context_dim: int | None = None,
        clamp_val: float = 10.0,
        residual: bool = False,
        orth_strategy: str = "cayley",
        add_self_loops: bool = True,
        add_eps: bool = False,
        sparse_learner: bool = False,
    ):
        """Initialize the layer.

        Args:
            stalk_dim: Stalk dimension (restriction maps are stalk_dim x stalk_dim
                orthogonal matrices).
            in_channels: Channel dimension f (W2 is fxf).
            hidden_dim: Projection width for attention and map-generator MLPs.
            num_heads: Independent attention heads (averaged after weighting).
            leaky_relu_slope: Negative slope for LeakyReLU in attention logits.
            context_dim: Flattened stalk size stalk_dim*in_channels; defaults to
                stalk_dim*in_channels.
            clamp_val: Clamping bound for Cayley transform inputs (ignored for fasth).
            residual: True -> Res-SheafAN (Eq. 6); False -> SheafAN (Eq. 5).
            orth_strategy: "cayley" or "fasth" (Householder). Defaults to "cayley".
            add_self_loops: If True, inserts self-loops before attention (GAT default).
                Set to False when the graph already contains them or they are unwanted.
            add_eps: With ``residual=True``, adds a learnable per-stalk rescaling of
                the residual term, ``(1 + tanh(eps)) * x_stalk``; zero-initialized.
            sparse_learner: Stalk-summed map-generator input, shrinking the learner
                from 2*final_d*hidden to 2*hidden inputs.
        """
        # W1, W2, sigma, reset_parameters all come from here. Attention has no
        # add_lp/add_hp counterpart: its softmax weights replace the degree
        # normalization the fixed channels are defined against.
        super().__init__(
            stalk_dim,
            in_channels,
            hidden_dim,
            context_dim=context_dim,
            sparse_learner=sparse_learner,
        )

        self.clamp_val = clamp_val
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual
        self.orth_strategy = orth_strategy
        self.add_self_loops = add_self_loops
        self.add_eps = add_eps
        if add_eps and residual:
            self.eps = nn.Parameter(torch.zeros(stalk_dim))

        # Replace Tanh (NSD default) with ELU.
        self.sigma = nn.ELU()

        # GAT-style attention: shared projection, split attention vector a.
        # a[:, :hidden_dim] weights the destination; a[:, hidden_dim:] the source.
        self.attn_projection = nn.Linear(self.context_dim, hidden_dim, bias=False)
        self.attn_score_vec = nn.Parameter(torch.empty(num_heads, 2 * hidden_dim))

        # One orthogonal map generator per head; parameter count depends on
        # the orthogonal parameterization (see orth_param_count).
        num_map_params = orth_param_count(orth_strategy, stalk_dim)
        self.map_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self._generator_in_dim(), hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_map_params),
                )
                for _ in range(num_heads)
            ]
        )

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()  # Xavier for W1, W2, and map_generators
        nn.init.xavier_uniform_(self.attn_score_vec)
        nn.init.xavier_uniform_(self.attn_projection.weight)

    def forward(
        self,
        x_feat: torch.Tensor,
        x_stalk: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run one sheaf attention layer.

        Args:
            x_feat: [N, context_dim] Flattened stalk features.
            x_stalk: [N, stalk_dim, in_channels] Structured stalk features to update.
            edge_index: [2, E] Graph connectivity (source -> dest).

        Returns:
            Updated stalk features [N, stalk_dim, in_channels].
        """
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x_stalk.size(0))

        # z = W1 @ x_stalk @ W2  [N, stalk_dim, in_channels]
        z = self._apply_stalk_transform(x_stalk)

        # Project node features to attention space [N, hidden_dim].
        # Split into destination and source logit contributions [N, num_heads]:
        #   a^T [h_i || h_j] = a_dst^T h_i + a_src^T h_j
        # PyG auto-indexes these per-node tensors to edge level inside message().
        node_proj = self.attn_projection(x_feat)
        attn_dst = node_proj @ self.attn_score_vec[:, : self.hidden_dim].T
        attn_src = node_proj @ self.attn_score_vec[:, self.hidden_dim :].T

        # Pre-index z to edge level (source side only; message() applies transport).
        # size passed explicitly so PyG aggregates to the correct node count.
        num_nodes = x_stalk.size(0)
        z_src = z[edge_index[0]]

        # All edge-level work (attention, transport, weighting) is done in message().
        propagated = self.propagate(  # ty: ignore[missing-argument]  PyG dispatch
            edge_index,  # ty: ignore[invalid-argument-type]
            z_src=z_src,
            x_feat=x_feat,
            attn_dst=attn_dst,
            attn_src=attn_src,
            size=(num_nodes, num_nodes),
        )

        # Res-SheafAN (Eq. 6): skip + sigma((A_F_hat - I) z)
        if self.residual:
            skip = (
                (1 + torch.tanh(self.eps)[None, :, None]) * x_stalk
                if self.add_eps
                else x_stalk
            )
            return skip + self.sigma(propagated - z)

        # SheafAN (Eq. 5): sigma(A_F_hat z)
        return self.sigma(propagated)

    def _to_ortho(self, map_params: torch.Tensor) -> torch.Tensor:
        return build_orthogonal(
            map_params, self.stalk_dim, self.orth_strategy, self.clamp_val
        )

    def message(  # ty: ignore[invalid-method-override]  PyG dynamic dispatch
        self,
        z_src: torch.Tensor,
        x_feat_i: torch.Tensor,
        x_feat_j: torch.Tensor,
        attn_dst_i: torch.Tensor,
        attn_src_j: torch.Tensor,
        index: torch.Tensor,
        size_i: int,
    ) -> torch.Tensor:
        r"""Attention-weighted orthogonal transport message for each edge.

        PyG auto-expands node tensors to edge level: x_feat -> x_feat_i / x_feat_j,
        attn_dst -> attn_dst_i, attn_src -> attn_src_j.

        Self-loops need no special handling: for edge :math:`(i, i)` both contexts
        are :math:`[x_{\text{feat},i} \,\|\, x_{\text{feat},i}]`, the two Cayley
        calls receive the same input, ``restriction_map_dst = restriction_map_src``,
        and :math:`P_{ii} = F^\top F = I` by orthogonality.
        """
        # Row-stochastic attention weights per destination node [E+N, num_heads].
        attn_weights = softmax(
            self.leaky_relu(attn_dst_i + attn_src_j), index, num_nodes=size_i
        )

        # Orthogonal transport P_ij = F_i^T F_j per head
        # [E+N, stalk_dim, stalk_dim].
        # Each map generator receives the concatenated features of both endpoints,
        # with the target node's features first to encode edge directionality.
        restr_map_input_i = torch.cat([x_feat_i, x_feat_j], dim=-1)
        restr_map_input_j = torch.cat([x_feat_j, x_feat_i], dim=-1)
        if self.sparse_learner:
            # Stalk-summed learner input (see BaseSheafConv._bidirectional_input).
            restr_map_input_i = restr_map_input_i.view(
                -1, self.stalk_dim, 2 * self.in_channels
            ).sum(dim=1)
            restr_map_input_j = restr_map_input_j.view(
                -1, self.stalk_dim, 2 * self.in_channels
            ).sum(dim=1)

        transport_maps = torch.stack(
            [
                torch.matmul(
                    self._to_ortho(gen(restr_map_input_i)).transpose(-2, -1),
                    self._to_ortho(gen(restr_map_input_j)),
                )
                for gen in self.map_generators
            ],
            dim=1,
        )  # [E+N, num_heads, stalk_dim, stalk_dim]

        # Average over heads after weighting by attention.
        weighted_transport_maps = (
            attn_weights.unsqueeze(-1).unsqueeze(-1) * transport_maps
        ).mean(dim=1)  # [E+N, stalk_dim, stalk_dim]

        return torch.matmul(weighted_transport_maps, z_src)


__all__ = ["SheafAttnConv"]
