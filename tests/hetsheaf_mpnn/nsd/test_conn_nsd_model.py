# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries.models import model_registry
from hetsheaf_mpnn.nsd import HetConnectionLaplacianNSDModel
from sheaf_mpnn.nsd import ConnectionLaplacianNSDModel

NUM_NODES = 12
IN_CHANNELS = 8
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4
_REG = RegConfig()


def _graph():
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


def _model() -> HetConnectionLaplacianNSDModel:
    return HetConnectionLaplacianNSDModel(
        IN_CHANNELS, OUT_CHANNELS, stalk_dim=3, hidden_dim=4, num_layers=2
    )


def test_construction_and_forward():
    torch.manual_seed(42)
    edge_index = _graph()
    nt, et = _types(edge_index)
    out = _model()(torch.randn(NUM_NODES, IN_CHANNELS), edge_index, nt, et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()


def test_types_do_not_change_output():
    """Feature-derived maps are type-blind: outputs match with and without types."""
    torch.manual_seed(0)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = _model().eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, edge_index), model(x, edge_index, nt, et))


def test_matches_homogeneous_model():
    torch.manual_seed(0)
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    het = _model().eval()
    homo = ConnectionLaplacianNSDModel(
        IN_CHANNELS, OUT_CHANNELS, stalk_dim=3, hidden_dim=4, num_layers=2
    ).eval()
    homo.load_state_dict(het.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(het(x, edge_index, nt, et), homo(x, edge_index))


def test_gradient_backprop():
    torch.manual_seed(0)
    edge_index = _graph()
    nt, et = _types(edge_index)
    model = _model()
    model(torch.randn(NUM_NODES, IN_CHANNELS), edge_index, nt, et).sum().backward()
    assert model.encoder.weight.grad is not None
    assert model.encoder.weight.grad.abs().sum() > 0


def test_in_channels_lt_stalk_dim_raises():
    with pytest.raises(ValueError, match="must be >= stalk_dim"):
        HetConnectionLaplacianNSDModel(in_channels=1, out_channels=3, stalk_dim=4)


class TestHetConnNSDRegistry:
    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.HET_CONN_NSD) == "het_conn_nsd"
        assert "het_conn_nsd" in model_registry

    def test_build_ignores_het_context_flags(self):
        # The flags have no generator to feed; building must not depend on them.
        cfg = ModelConfig(
            type=ModelType.HET_CONN_NSD,
            stalk_dim=3,
            hidden_dim=4,
            num_layers=1,
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            use_node_types=True,
            use_edge_types=True,
        )
        model = model_registry.build(
            "het_conn_nsd", IN_CHANNELS, OUT_CHANNELS, cfg, _REG
        )
        assert isinstance(model, HetConnectionLaplacianNSDModel)
        edge_index = _graph()
        nt, et = _types(edge_index)
        out = model(torch.randn(NUM_NODES, IN_CHANNELS), edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
