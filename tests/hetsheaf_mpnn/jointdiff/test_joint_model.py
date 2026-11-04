# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from hetsheaf_mpnn.jointdiff import (
    HetJointDiffusionModel,
    HetJointDiffusionSheafConv,
    HetRotationInvariantModel,
)

NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_construction_and_forward(model_cls):
    torch.manual_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)

    model = model_cls(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_forward_without_types(model_cls):
    """Model works without type tensors (features-only mode)."""
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = model_cls(
        IN_CHANNELS,
        OUT_CHANNELS,
        use_features=True,
        use_node_types=False,
        use_edge_types=False,
    )
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_ensemble_model(model_cls):
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    model = model_cls(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        ensemble=True,
    )
    out = model(x, edge_index, None, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_gradient_backprop(model_cls):
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = model_cls(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=True,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    out.sum().backward()
    assert model.encoder.weight.grad is not None


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_jknet(model_cls):
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = model_cls(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=3,
        jknet=True,
        normalize_output=True,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_joint_only_first_layer_owns_initializer():
    model = HetJointDiffusionModel(IN_CHANNELS, OUT_CHANNELS, num_layers=3)
    layers = list(model.layers)
    assert layers[0].learn_init_maps
    assert hasattr(layers[0], "map_generator")
    for layer in layers[1:]:
        assert not layer.learn_init_maps
        assert not hasattr(layer, "map_generator")


def test_joint_layer_without_prev_maps_raises():
    layer = HetJointDiffusionSheafConv(
        stalk_dim=2, in_channels=4, hidden_dim=8, learn_init_maps=False
    )
    edge_index = _graph()
    x_stalk = torch.randn(NUM_NODES, 2, 4)
    x_feat = x_stalk.reshape(NUM_NODES, -1)
    with pytest.raises(RuntimeError, match="learn_init_maps=False"):
        layer(x_feat, x_stalk, edge_index)


def test_joint_maps_evolve_across_layers():
    """The maps returned by a layer differ from the ones it received."""
    torch.manual_seed(0)
    edge_index = _graph()
    layer0 = HetJointDiffusionSheafConv(stalk_dim=2, in_channels=4, hidden_dim=8)
    layer1 = HetJointDiffusionSheafConv(
        stalk_dim=2, in_channels=4, hidden_dim=8, learn_init_maps=False
    )
    x_stalk = torch.randn(NUM_NODES, 2, 4)
    x_feat = x_stalk.reshape(NUM_NODES, -1)
    x_stalk, maps0 = layer0(x_feat, x_stalk, edge_index)
    x_feat = x_stalk.reshape(NUM_NODES, -1)
    _, maps1 = layer1(x_feat, x_stalk, edge_index, prev_maps=maps0)
    assert maps0[0].shape == (edge_index.size(1), 2, 2)
    assert not torch.allclose(maps0[0], maps1[0])


def test_rot_inv_time_dep_threading():
    """time_dep=True feeds prev_maps into the next layer without erroring."""
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetRotationInvariantModel(
        IN_CHANNELS, OUT_CHANNELS, num_layers=3, time_dep=True
    )
    out = model(x, edge_index)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_rot_inv_types_only():
    """use_features=False with node types yields a type-driven sheaf."""
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetRotationInvariantModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_features=False,
        use_node_types=True,
        use_edge_types=True,
    )
    out = model(x, edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_invalid_args(model_cls):
    with pytest.raises(ValueError, match="stalk_dim"):
        model_cls(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        model_cls(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)
    with pytest.raises(ValueError, match="at least one"):
        model_cls(IN_CHANNELS, OUT_CHANNELS, num_layers=0)


@pytest.mark.parametrize(
    "model_cls", [HetJointDiffusionModel, HetRotationInvariantModel]
)
def test_reset_parameters(model_cls):
    model = model_cls(IN_CHANNELS, OUT_CHANNELS, num_node_types=2, num_edge_types=2)
    model.reset_parameters()
