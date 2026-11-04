# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the BSNN convolution layers (sampled-map NSD diffusion)."""

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.bsnn.bsnn_layers import (
    DiagonalBSNNConv,
    GeneralBSNNConv,
    LowRankBSNNConv,
    OrthogonalBSNNConv,
)
from sheaf_mpnn.bsnn.variational import sample_uniform_so

_LAYERS = {
    "diagonal": DiagonalBSNNConv,
    "general": GeneralBSNNConv,
    "low_rank": LowRankBSNNConv,
    "orthogonal": OrthogonalBSNNConv,
}

# The default rank of LowRankBSNNConv, matched by _make_maps' factor width.
_DEFAULT_RANK = 1


def _make_maps(kind: str, num_edges: int, d: int) -> torch.Tensor:
    torch.manual_seed(1)
    if kind == "diagonal":
        return torch.randn(2, num_edges, d)
    if kind == "general":
        return torch.randn(2, num_edges, d, d)
    if kind == "low_rank":
        return torch.randn(2, num_edges, d, 2 * _DEFAULT_RANK)
    return torch.stack(
        [sample_uniform_so(num_edges, d), sample_uniform_so(num_edges, d)]
    )


@pytest.mark.parametrize("kind", list(_LAYERS))
class TestBSNNLayers:
    @pytest.fixture
    def setup(self, kind):
        torch.manual_seed(42)
        num_nodes, d, f = 12, 3, 4
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        x_stalk = torch.randn(num_nodes, d, f)
        layer = _LAYERS[kind](stalk_dim=d, in_channels=f, hidden_dim=f)
        maps = _make_maps(kind, edge_index.size(1), d)
        return layer, maps, x_stalk, edge_index

    def test_forward_shape(self, setup, kind):
        layer, maps, x_stalk, edge_index = setup
        out = layer(maps, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_gradients_flow_to_weights_and_maps(self, setup, kind):
        layer, maps, x_stalk, edge_index = setup
        maps = maps.clone().requires_grad_(True)
        out = layer(maps, x_stalk, edge_index)
        out.pow(2).mean().backward()
        assert layer.W1.grad is not None and layer.W1.grad.abs().sum() > 0
        assert layer.W2.grad is not None and layer.W2.grad.abs().sum() > 0
        assert maps.grad is not None and maps.grad.abs().sum() > 0

    def test_empty_graph_stability(self, setup, kind):
        layer, _, x_stalk, _ = setup
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        maps = _make_maps(kind, 0, x_stalk.size(1))
        out = layer(maps, x_stalk, empty_edges)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_fixed_channels_extend_stalk(self, kind):
        torch.manual_seed(42)
        num_nodes, d, f = 10, 2, 4
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        layer = _LAYERS[kind](
            stalk_dim=d, in_channels=f, hidden_dim=f, add_lp=True, add_hp=True
        )
        assert layer.stalk_dim == d + 2
        x_stalk = torch.randn(num_nodes, d + 2, f)
        maps = _make_maps(kind, edge_index.size(1), d)
        out = layer(maps, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_rejects_self_loops(self, kind):
        with pytest.raises(ValueError, match="self-loops"):
            _LAYERS[kind](stalk_dim=3, in_channels=4, hidden_dim=4, add_self_loops=True)


class TestLowRankBSNNConv:
    def test_products_match_general_conv_on_composed_maps(self):
        """The factored Gram products must equal what GeneralBSNNConv computes
        on the composed maps F = A B^T -- the factorization is an
        implementation detail, not a different operator.
        """
        torch.manual_seed(3)
        num_nodes, d, f, r = 10, 3, 4, 2
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
        factors = torch.randn(2, edge_index.size(1), d, 2 * r)
        left, right = factors.chunk(2, dim=-1)
        composed = left @ right.transpose(-2, -1)  # [2, E, d, d]

        low_rank = LowRankBSNNConv(stalk_dim=d, in_channels=f, hidden_dim=f, rank=r)
        general = GeneralBSNNConv(stalk_dim=d, in_channels=f, hidden_dim=f)
        self_lr, cross_lr = low_rank.get_map_products(factors, edge_index)
        self_gen, cross_gen = general.get_map_products(composed, edge_index)
        torch.testing.assert_close(self_lr, self_gen)
        torch.testing.assert_close(cross_lr, cross_gen)

    def test_products_are_rank_bounded(self):
        torch.manual_seed(4)
        num_nodes, d, f, r = 10, 4, 3, 1
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
        factors = torch.randn(2, edge_index.size(1), d, 2 * r)
        layer = LowRankBSNNConv(stalk_dim=d, in_channels=f, hidden_dim=f, rank=r)
        _, cross_map = layer.get_map_products(factors, edge_index)
        assert (torch.linalg.matrix_rank(cross_map) <= r).all()

    def test_forward_with_rank_above_one(self):
        torch.manual_seed(5)
        num_nodes, d, f, r = 12, 3, 4, 2
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        layer = LowRankBSNNConv(stalk_dim=d, in_channels=f, hidden_dim=f, rank=r)
        x_stalk = torch.randn(num_nodes, d, f)
        maps = torch.randn(2, edge_index.size(1), d, 2 * r)
        out = layer(maps, x_stalk, edge_index)
        assert out.shape == x_stalk.shape
        assert torch.isfinite(out).all()

    def test_mismatched_factor_width_is_rejected(self):
        torch.manual_seed(6)
        num_nodes, d, f = 8, 3, 4
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
        layer = LowRankBSNNConv(stalk_dim=d, in_channels=f, hidden_dim=f, rank=2)
        bad_maps = torch.randn(2, edge_index.size(1), d, 2)  # rank-1 width
        with pytest.raises(ValueError, match="trailing dimension"):
            layer(bad_maps, torch.randn(num_nodes, d, f), edge_index)

    def test_rejects_non_positive_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            LowRankBSNNConv(stalk_dim=3, in_channels=4, hidden_dim=4, rank=0)


def test_orthogonal_identity_maps_reduce_to_graph_laplacian_action():
    """With identity restriction maps everywhere, the cross-map product is the
    identity, so the layer's Laplacian action must be identical across the
    stalk dimensions (pure graph diffusion in each channel).
    """
    torch.manual_seed(0)
    num_nodes, d, f = 8, 2, 3
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    eye = torch.eye(d).expand(edge_index.size(1), d, d)
    maps = torch.stack([eye, eye])
    layer = OrthogonalBSNNConv(stalk_dim=d, in_channels=f, hidden_dim=f)
    with torch.no_grad():
        layer.W1.copy_(torch.eye(d))
        layer.W2.copy_(torch.eye(f))
    x_stalk = torch.randn(num_nodes, 1, f).expand(num_nodes, d, f).contiguous()
    out = layer(maps, x_stalk, edge_index)
    # Identical per-channel inputs + channel-agnostic diffusion => identical
    # per-channel outputs.
    torch.testing.assert_close(out[:, 0], out[:, 1])
