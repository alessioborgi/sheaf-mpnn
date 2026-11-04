"""Tests for the Rotation Invariant Sheaf model (RotationInvariantModel)."""

import random

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.jointdiff.joint_model import RotationInvariantModel
from sheaf_mpnn.jointdiff.rot_inv_layers import RotationInvariantSheafConv


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def setup_rotinv():
    set_seed(123)
    num_nodes = 7
    stalk_dim = 4
    hidden_dim = 6
    in_channels = 10
    out_channels = 4
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.3)
    model = RotationInvariantModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=stalk_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        alpha=1.0,
    )
    x = torch.randn(num_nodes, in_channels)
    return model, x, edge_index


def test_forward_shape(setup_rotinv):
    model, x, edge_index = setup_rotinv
    out = model(x, edge_index)
    assert out.shape == (x.size(0), model.out_channels)
    assert not torch.isnan(out).any()


def test_gradient_flow(setup_rotinv):
    model, x, edge_index = setup_rotinv
    out = model(x, edge_index)
    loss = out.mean()
    loss.backward()
    assert model.encoder.weight.grad is not None
    assert model.encoder.weight.grad.abs().sum() > 0
    for layer in model.layers:
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None and param.grad.abs().sum() > 0
        break


def test_alpha_is_wired_and_freezable():
    """cfg.alpha must reach every layer; learn_alpha=False must freeze it."""
    model = RotationInvariantModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        alpha=2.0,
    )
    for layer in model.layers:
        assert isinstance(layer, RotationInvariantSheafConv)
        assert layer.alpha.item() == pytest.approx(2.0)

    frozen = RotationInvariantModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        learn_alpha=False,
    )
    for layer in frozen.layers:
        assert not layer.alpha.requires_grad


def test_time_dep_toggle():
    """time_dep must reach the layers; both modes must run forward."""
    set_seed(7)
    edge_index = erdos_renyi_graph(6, edge_prob=0.5)
    x = torch.randn(6, 8)
    for time_dep in (True, False):
        model = RotationInvariantModel(
            in_channels=8,
            out_channels=2,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=3,
            time_dep=time_dep,
        )
        for layer in model.layers:
            assert layer.time_dep is time_dep
        out = model(x, edge_index)
        assert out.shape == (6, 2)
        assert torch.isfinite(out).all()


def test_forward_with_isolated_nodes():
    """Propagate must size the aggregation to num_nodes, not max edge id."""
    num_nodes = 6
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # nodes 3-5 isolated
    model = RotationInvariantModel(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    out = model(torch.randn(num_nodes, 8), edge_index)
    assert out.shape == (num_nodes, 2)
    assert torch.isfinite(out).all()
