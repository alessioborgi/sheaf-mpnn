"""Tests for the Joint Diffusion Sheaf model (JointDiffusionModel)."""

import random
from typing import Any

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from sheaf_mpnn.jointdiff.joint_layers import JointDiffusionSheafConv
from sheaf_mpnn.jointdiff.joint_model import JointDiffusionModel, RotationInvariantModel


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@pytest.fixture
def setup_jointdiff():
    set_seed(42)
    num_nodes = 8
    stalk_dim = 4
    hidden_dim = 8
    in_channels = 16
    out_channels = 3
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = JointDiffusionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=stalk_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        beta=1.0,
        alpha=1.0,
        add_self_loops=False,
        add_eps=True,
        input_dropout=0.0,
        dropout=0.0,
        normalize_output=False,
        jknet=False,
        learn_alpha=True,
    )
    x = 10 * torch.rand(num_nodes, in_channels)
    return model, x, edge_index


def test_forward_shape(setup_jointdiff):
    model, x, edge_index = setup_jointdiff
    out = model(x, edge_index)
    assert out.shape == (x.size(0), model.out_channels)
    assert not torch.isnan(out).any()


def test_gradient_flow(setup_jointdiff):
    model, x, edge_index = setup_jointdiff
    out = model(x, edge_index)
    loss = out.pow(2).sum()
    loss.backward()
    assert model.encoder.weight.grad is not None
    assert model.encoder.weight.grad.abs().sum() > 0
    assert model.decoder.weight.grad is not None
    assert model.decoder.weight.grad.abs().sum() > 0
    for param in model.layers[0].init_map_generator.parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0
    for layer in model.layers:
        assert layer.alpha.grad is not None
        assert layer.Wf1.grad is not None and layer.Wf1.grad.abs().sum() > 0
        assert layer.Wf2.grad is not None and layer.Wf2.grad.abs().sum() > 0


def test_only_first_layer_owns_init_generator():
    """Layers after the first receive prev_maps, so a per-layer generator
    would be dead weight; only layer 0 may own one.
    """
    model = JointDiffusionModel(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=3
    )
    layers = list(model.layers)
    assert isinstance(layers[0], JointDiffusionSheafConv)
    assert layers[0].init_map_generator is not None
    for layer in layers[1:]:
        assert isinstance(layer, JointDiffusionSheafConv)
        assert layer.init_map_generator is None


def test_alpha_is_wired_and_freezable():
    """cfg.alpha must reach every layer and survive reset_parameters;
    learn_alpha=False must freeze it.
    """
    model = JointDiffusionModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        alpha=0.5,
    )
    for layer in model.layers:
        assert isinstance(layer, JointDiffusionSheafConv)
        assert layer.alpha.item() == pytest.approx(0.5)
    model.reset_parameters()
    for layer in model.layers:
        assert isinstance(layer, JointDiffusionSheafConv)
        assert layer.alpha.item() == pytest.approx(0.5)

    frozen = JointDiffusionModel(
        in_channels=8,
        out_channels=2,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        alpha=0.5,
        learn_alpha=False,
    )
    for layer in frozen.layers:
        assert not layer.alpha.requires_grad


def test_forward_with_isolated_nodes():
    """Propagate must size the aggregation to num_nodes, not max edge id."""
    num_nodes = 6
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # nodes 3-5 isolated
    model = JointDiffusionModel(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    out = model(torch.randn(num_nodes, 8), edge_index)
    assert out.shape == (num_nodes, 2)
    assert torch.isfinite(out).all()


def test_jknet_output_consistency():
    """Enables JKNet and checks output shape and finiteness."""
    set_seed(0)
    num_nodes = 6
    stalk_dim = 3
    hidden_dim = 5
    in_channels = 12
    out_channels = 2
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
    model = JointDiffusionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=stalk_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        jknet=True,
    )
    x = torch.randn(num_nodes, in_channels)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, out_channels)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "model_cls", [JointDiffusionModel, RotationInvariantModel], ids=["joint", "rotinv"]
)
@pytest.mark.parametrize(
    "bad_kwargs,match",
    [
        ({"stalk_dim": 0}, "stalk_dim must be positive"),
        ({"hidden_dim": 0}, "hidden_dim must be positive"),
        ({"num_layers": 0}, "must have at least one layer"),
    ],
)
def test_invalid_constructor_args_raise(model_cls, bad_kwargs, match):
    """Non-positive stalk_dim/hidden_dim/num_layers must be rejected."""
    kwargs: dict[str, Any] = dict(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        model_cls(**kwargs)


@pytest.mark.parametrize(
    "model_cls", [JointDiffusionModel, RotationInvariantModel], ids=["joint", "rotinv"]
)
def test_forward_rejects_bad_inputs(model_cls):
    """Forward must validate the shapes of x and edge_index."""
    model = model_cls(
        in_channels=8, out_channels=2, stalk_dim=2, hidden_dim=4, num_layers=2
    )
    edge_index = torch.tensor([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="x must have shape"):
        model(torch.randn(4, 7), edge_index)
    with pytest.raises(ValueError, match="x must have shape"):
        model(torch.randn(8), edge_index)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        model(torch.randn(4, 8), torch.zeros(3, 2, dtype=torch.long))


@pytest.mark.parametrize(
    "model_cls", [JointDiffusionModel, RotationInvariantModel], ids=["joint", "rotinv"]
)
def test_second_linear_and_reset_parameters(model_cls):
    """second_linear adds encoder2 to both forward and reset_parameters; a
    reset must re-initialize the weights and change the output.
    """
    set_seed(7)
    num_nodes = 6
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = model_cls(
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


@pytest.mark.parametrize(
    "model_cls", [JointDiffusionModel, RotationInvariantModel], ids=["joint", "rotinv"]
)
def test_jknet_with_normalized_output(model_cls):
    """JKNet with normalize_output must normalize each layer rep and the JK
    concatenation while preserving the output shape.
    """
    set_seed(3)
    num_nodes = 6
    in_channels = 12
    out_channels = 2
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = model_cls(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        jknet=True,
        normalize_output=True,
    )
    out = model(torch.randn(num_nodes, in_channels), edge_index)
    assert out.shape == (num_nodes, out_channels)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "model_cls", [JointDiffusionModel, RotationInvariantModel], ids=["joint", "rotinv"]
)
def test_normalized_output_without_jknet(model_cls):
    """normalize_output alone must normalize the final representation."""
    set_seed(4)
    num_nodes = 6
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.5)
    model = model_cls(
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


def test_eval_forward_is_deterministic(setup_jointdiff):
    """In eval mode two forward passes must produce identical outputs."""
    model, x, edge_index = setup_jointdiff
    model.eval()
    with torch.no_grad():
        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
    assert torch.allclose(out1, out2)
