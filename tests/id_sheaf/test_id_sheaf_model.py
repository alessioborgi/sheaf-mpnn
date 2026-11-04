"""Tests for the Identity Sheaf model (IdentitySheafModel)."""

import random
from typing import Any

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.id_sheaf.id_sheaf_layers import IdentitySheafConv
from sheaf_mpnn.id_sheaf.id_sheaf_model import IdentitySheafModel


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def setup_identity():
    set_seed(2026)
    num_nodes = 9
    stalk_dim = 4
    hidden_dim = 8
    in_channels = 16
    out_channels = 5
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.45)
    model = IdentitySheafModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=stalk_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        alpha=1.0,
    )
    x = torch.randn(num_nodes, in_channels)
    return model, x, edge_index


def test_forward_shape(setup_identity):
    model, x, edge_index = setup_identity
    out = model(x, edge_index)
    assert out.shape == (x.size(0), model.out_channels)
    assert not torch.isnan(out).any()


def test_gradient_flow(setup_identity):
    model, x, edge_index = setup_identity
    out = model(x, edge_index)
    loss = out.pow(2).sum()
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
    model = IdentitySheafModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        alpha=0.25,
    )
    for layer in model.layers:
        assert isinstance(layer, IdentitySheafConv)
        assert layer.alpha.item() == pytest.approx(0.25)

    frozen = IdentitySheafModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        learn_alpha=False,
    )
    for layer in frozen.layers:
        assert not layer.alpha.requires_grad


def test_laplacian_is_degree_normalized():
    """With identity maps the products must be scaled by (D+1)^{-1/2} like
    every other family, so messages do not grow with node degree.
    """
    layer = IdentitySheafConv(stalk_dim=2, in_channels=4, hidden_dim=4)
    # Directed triangle plus an extra edge into node 0: deg_in = [2, 1, 1].
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 0]])
    num_nodes = 3
    x_feat = torch.randn(num_nodes, 8)

    self_map, cross_map = layer.get_map_products(x_feat, edge_index)
    norm_self, norm_cross = layer._apply_norm(
        self_map, cross_map, edge_index, num_nodes
    )

    deg_in = torch.tensor([2.0, 1.0, 1.0])
    inv_sqrt = (deg_in + 1).pow(-0.5)
    dst, src = edge_index[1], edge_index[0]
    expected_self = (inv_sqrt[dst] ** 2)[:, None].expand_as(norm_self)
    expected_cross = (inv_sqrt[dst] * inv_sqrt[src])[:, None].expand_as(norm_cross)
    assert torch.allclose(norm_self, expected_self)
    assert torch.allclose(norm_cross, expected_cross)


def test_forward_with_isolated_nodes():
    """Propagate must size the aggregation to num_nodes, not max edge id."""
    num_nodes = 6
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # nodes 3-5 isolated
    model = IdentitySheafModel(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    out = model(torch.randn(num_nodes, 8), edge_index)
    assert out.shape == (num_nodes, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "bad_kwargs,match",
    [
        ({"stalk_dim": 0}, "stalk_dim must be positive"),
        ({"hidden_dim": 0}, "hidden_dim must be positive"),
        ({"num_layers": 0}, "must have at least one layer"),
    ],
)
def test_invalid_constructor_args_raise(bad_kwargs, match):
    """Non-positive stalk_dim/hidden_dim/num_layers must be rejected."""
    kwargs: dict[str, Any] = dict(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        IdentitySheafModel(**kwargs)


def test_forward_rejects_bad_inputs():
    """Forward must validate the shapes of x and edge_index."""
    model = IdentitySheafModel(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    edge_index = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="x must have shape"):
        model(torch.randn(4, 7), edge_index)
    with pytest.raises(ValueError, match="x must have shape"):
        model(torch.randn(8), edge_index)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(torch.randn(4, 8), torch.zeros(3, 2, dtype=torch.long))


def test_second_linear_and_reset_parameters():
    """second_linear adds encoder2 to both forward and reset_parameters; a
    reset must re-initialize the weights and change the output.
    """
    set_seed(7)
    num_nodes = 6
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = IdentitySheafModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        second_linear=True,
    )
    assert model.encoder2 is not None
    x = torch.randn(num_nodes, 8)
    model.eval()
    with torch.no_grad():
        out1 = model(x, edge_index)
    model.reset_parameters()
    with torch.no_grad():
        out2 = model(x, edge_index)
    assert out1.shape == out2.shape == (num_nodes, 2)
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()
    assert not torch.allclose(out1, out2)


def test_jknet_with_normalized_output():
    """JKNet with normalize_output must normalize each layer rep and the JK
    concatenation while preserving the output shape.
    """
    set_seed(3)
    num_nodes = 6
    in_channels = 12
    out_channels = 2
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = IdentitySheafModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=3,
        jknet=True,
        normalize_output=True,
    )
    out = model(torch.randn(num_nodes, in_channels), edge_index)
    assert out.shape == (num_nodes, out_channels)
    assert torch.isfinite(out).all()


def test_normalized_output_without_jknet():
    """normalize_output alone must normalize the final representation."""
    set_seed(4)
    num_nodes = 6
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = IdentitySheafModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        normalize_output=True,
    )
    out = model(torch.randn(num_nodes, 8), edge_index)
    assert out.shape == (num_nodes, 2)
    assert torch.isfinite(out).all()


def test_eval_forward_is_deterministic(setup_identity):
    """In eval mode two forward passes must produce identical outputs."""
    model, x, edge_index = setup_identity
    model.eval()
    with torch.no_grad():
        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
    assert torch.allclose(out1, out2)
