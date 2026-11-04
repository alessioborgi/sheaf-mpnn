# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import add_self_loops, to_undirected

from sheaf_mpnn.nsp import ConnectionLaplacianNSPConv, ConnectionLaplacianNSPModel
from sheaf_mpnn.utils.connection_laplacian import (
    _ensure_min_degree,
    _local_pca,
    _orthogonal_alignment,
)


def _make_maps(num_nodes, feat_dim, stalk_dim, edge_index):
    """Replicate the model's edge preprocessing: symmetrise, augment, add self-loops."""
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = _ensure_min_degree(
        torch.randn(num_nodes, feat_dim), edge_index, stalk_dim
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    x = torch.randn(num_nodes, feat_dim)
    bases = _local_pca(x, edge_index, stalk_dim)
    maps = _orthogonal_alignment(bases, edge_index)
    return maps, edge_index


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------


class TestConnectionLaplacianNSPConv:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        num_nodes, feat_dim, stalk_dim, hidden_dim = 10, 8, 3, 4
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]],
            dtype=torch.long,
        )
        maps, edge_index = _make_maps(num_nodes, feat_dim, stalk_dim, edge_index)
        # NSP augmented stalk: [N, 2d, f]
        x_stalk = torch.randn(num_nodes, 2 * stalk_dim, hidden_dim)
        conv = ConnectionLaplacianNSPConv(
            stalk_dim=stalk_dim,
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            add_self_loops=False,  # edge_index from _make_maps already has self-loops
        )
        return conv, maps, x_stalk, edge_index, stalk_dim, hidden_dim

    def test_output_shape(self, setup):
        conv, maps, x_stalk, edge_index, stalk_dim, hidden_dim = setup
        out = conv(maps, x_stalk, edge_index)
        assert out.shape == x_stalk.shape

    def test_no_nan(self, setup):
        conv, maps, x_stalk, edge_index, *_ = setup
        out = conv(maps, x_stalk, edge_index)
        assert torch.isfinite(out).all()

    def test_history_update(self, setup):
        conv, maps, x_stalk, edge_index, stalk_dim, hidden_dim = setup
        x_curr_before = x_stalk[:, :stalk_dim, :].clone()
        out = conv(maps, x_stalk, edge_index)
        x_curr_after = out[:, stalk_dim:, :]
        # Second half of output is the previous x_curr (NSP state augmentation).
        torch.testing.assert_close(x_curr_after, x_curr_before)

    def test_gradient_flows(self, setup):
        conv, maps, x_stalk, edge_index, *_ = setup
        out = conv(maps, x_stalk, edge_index)
        out.pow(2).sum().backward()
        assert conv.W1.grad is not None and conv.W1.grad.abs().sum() > 0

    def test_no_map_generator(self, setup):
        conv, *_ = setup
        assert not hasattr(conv, "map_generator")


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestConnectionLaplacianNSPModel:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        num_nodes, in_channels, out_channels = 12, 8, 3
        stalk_dim, hidden_dim, num_layers = 3, 4, 2
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
            ],
            dtype=torch.long,
        )
        x = torch.randn(num_nodes, in_channels)
        model = ConnectionLaplacianNSPModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        return model, x, edge_index, num_nodes, out_channels

    def test_output_shape(self, setup):
        model, x, edge_index, num_nodes, out_channels = setup
        out = model(x, edge_index)
        assert out.shape == (num_nodes, out_channels)

    def test_no_nan(self, setup):
        model, x, edge_index, *_ = setup
        out = model(x, edge_index)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, setup):
        model, x, edge_index, *_ = setup
        out = model(x, edge_index)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum() > 0 for g in grads)

    def test_in_channels_lt_stalk_dim_raises(self):
        with pytest.raises(ValueError, match="in_channels"):
            ConnectionLaplacianNSPModel(in_channels=2, out_channels=3, stalk_dim=4)

    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_various_num_layers(self, num_layers):
        torch.manual_seed(0)
        x = torch.randn(12, 8)
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
            ],
            dtype=torch.long,
        )
        model = ConnectionLaplacianNSPModel(
            in_channels=8, out_channels=3, stalk_dim=3, num_layers=num_layers
        )
        out = model(x, edge_index)
        assert out.shape == (12, 3)
        assert torch.isfinite(out).all()

    def test_stalk_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="stalk_dim must be positive"):
            ConnectionLaplacianNSPModel(in_channels=8, out_channels=3, stalk_dim=0)

    def test_hidden_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            ConnectionLaplacianNSPModel(
                in_channels=8, out_channels=3, stalk_dim=3, hidden_dim=0
            )

    def test_num_layers_must_be_positive(self):
        with pytest.raises(ValueError, match="at least one layer"):
            ConnectionLaplacianNSPModel(
                in_channels=8, out_channels=3, stalk_dim=3, num_layers=0
            )

    def test_reset_parameters_changes_output(self, setup):
        model, x, edge_index, *_ = setup
        model.eval()
        before = model(x, edge_index).clone()
        model.reset_parameters()
        assert not torch.allclose(before, model(x, edge_index))

    def test_add_self_loops_forward(self, setup):
        _, x, edge_index, num_nodes, out_channels = setup
        torch.manual_seed(0)
        model = ConnectionLaplacianNSPModel(
            in_channels=8, out_channels=3, stalk_dim=3, add_self_loops=True
        ).eval()
        out = model(x, edge_index)
        assert out.shape == (num_nodes, out_channels)
        assert torch.isfinite(out).all()
