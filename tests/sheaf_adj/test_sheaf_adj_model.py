# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.sheaf_adj.sheaf_adj_model import SheafAdjModel, SheafAdjVariant
from tests.testing import set_seed


@pytest.mark.parametrize("variant", list(SheafAdjVariant))
class TestSheafAdjModel:
    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        in_channels, out_channels = 10, 3
        num_nodes = 15
        model = SheafAdjModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=2,
            variant=variant,
        )
        x = torch.randn(num_nodes, in_channels)
        edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
        return model, x, edge_index

    def test_model_forward_shape(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        assert out.shape == (x.size(0), model.out_channels)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()

    def test_model_gradient_flow(self, setup):
        model, x, edge_index = setup
        out = model(x, edge_index)
        loss = out.pow(2).sum()
        loss.backward()

        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        for layer in model.layers:
            assert layer.W1.grad is not None and layer.W1.grad.abs().sum() > 0

    def test_model_determinism(self, setup):
        model, x, edge_index = setup
        model.eval()
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_isolated_graph_stability(self, setup):
        model, x, _ = setup
        model.eval()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()


def test_model_invalid_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        SheafAdjModel(in_channels=10, out_channels=3, stalk_dim=0)


def test_model_invalid_num_layers():
    with pytest.raises(ValueError, match="must have at least one"):
        SheafAdjModel(in_channels=10, out_channels=3, num_layers=0)


def test_model_jknet_forward():
    set_seed(42)
    model = SheafAdjModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=3,
        jknet=True,
    )
    x = torch.randn(12, 10)
    edge_index = erdos_renyi_graph(12, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (12, 3)
    assert not torch.isnan(out).any()


def test_model_add_eps_forward():
    set_seed(42)
    model = SheafAdjModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        add_eps=True,
    )
    x = torch.randn(12, 10)
    edge_index = erdos_renyi_graph(12, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (12, 3)
    assert not torch.isnan(out).any()


def test_hidden_dim_must_be_positive():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        SheafAdjModel(in_channels=10, out_channels=3, hidden_dim=0)


def test_learn_alpha_false_freezes_alpha():
    model = SheafAdjModel(in_channels=10, out_channels=3, learn_alpha=False)
    assert all(not layer.alpha.requires_grad for layer in model.layers)


def test_reset_parameters_changes_output():
    set_seed(42)
    model = SheafAdjModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        second_linear=True,
        jknet=True,
    ).eval()
    x = torch.randn(15, 10)
    edge_index = erdos_renyi_graph(15, edge_prob=0.4)
    before = model(x, edge_index).clone()
    model.reset_parameters()
    assert not torch.allclose(before, model(x, edge_index))


def test_forward_validates_x_shape():
    model = SheafAdjModel(in_channels=10, out_channels=3)
    with pytest.raises(ValueError, match="x must have shape"):
        model(torch.randn(5, 7), torch.empty((2, 0), dtype=torch.long))


def test_forward_validates_edge_index_shape():
    model = SheafAdjModel(in_channels=10, out_channels=3)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(torch.randn(5, 10), torch.zeros((3, 4), dtype=torch.long))
