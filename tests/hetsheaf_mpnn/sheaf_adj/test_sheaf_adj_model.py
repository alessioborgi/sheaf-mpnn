# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries.models import model_registry
from hetsheaf_mpnn.sheaf_adj.sheaf_adj_model import HetSheafAdjModel, HetSheafAdjVariant
from tests.testing import set_seed

_REG = RegConfig()


@pytest.mark.parametrize("variant", list(HetSheafAdjVariant))
class TestHetSheafAdjModel:
    @pytest.fixture
    def setup(self, variant):
        set_seed(42)
        in_channels, out_channels = 10, 3
        num_nodes = 15
        model = HetSheafAdjModel(
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


def test_het_model_with_node_types():
    set_seed(42)
    num_nodes, in_channels, out_channels = 12, 8, 3
    num_node_types = 2
    model = HetSheafAdjModel(
        in_channels=in_channels,
        out_channels=out_channels,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        variant=HetSheafAdjVariant.GENERAL,
        num_node_types=num_node_types,
        use_node_types=True,
    )
    x = torch.randn(num_nodes, in_channels)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.4)
    node_types = torch.randint(0, num_node_types, (num_nodes,))
    out = model(x, edge_index, node_types=node_types)
    assert out.shape == (num_nodes, out_channels)
    assert not torch.isnan(out).any()


def test_het_model_invalid_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        HetSheafAdjModel(in_channels=10, out_channels=3, stalk_dim=0)


def test_het_model_invalid_num_layers():
    with pytest.raises(ValueError, match="must have at least one"):
        HetSheafAdjModel(in_channels=10, out_channels=3, num_layers=0)


def test_hidden_dim_must_be_positive():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        HetSheafAdjModel(in_channels=10, out_channels=3, hidden_dim=0)


def test_jknet_forward_and_reset():
    set_seed(42)
    model = HetSheafAdjModel(
        in_channels=10,
        out_channels=3,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        jknet=True,
        normalize_output=True,
    ).eval()
    x = torch.randn(15, 10)
    edge_index = erdos_renyi_graph(15, edge_prob=0.4)
    out = model(x, edge_index)
    assert out.shape == (15, 3)
    assert torch.isfinite(out).all()

    before = model(x, edge_index).clone()
    model.reset_parameters()
    assert not torch.allclose(before, model(x, edge_index))


class TestHetSheafAdjRegistry:
    _N, _IN, _OUT, _NT, _ET = 12, 8, 3, 3, 4

    def _cfg(self, **kwargs) -> ModelConfig:
        defaults = dict(
            type=ModelType.HET_SHEAF_ADJ,
            variant="general",
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            num_node_types=self._NT,
            num_edge_types=self._ET,
            use_node_types=True,
            use_edge_types=True,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]

    def _inputs(self):
        edge_index = erdos_renyi_graph(self._N, edge_prob=0.4)
        nt = torch.randint(0, self._NT, (self._N,))
        et = torch.randint(0, self._ET, (edge_index.size(1),))
        return torch.randn(self._N, self._IN), edge_index, nt, et

    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.HET_SHEAF_ADJ) == "het_sheaf_adj"
        assert "het_sheaf_adj" in model_registry

    @pytest.mark.parametrize("variant", [v.name.lower() for v in HetSheafAdjVariant])
    def test_build_returns_a_working_model(self, variant):
        cfg = self._cfg(variant=variant, rank=2)
        model = model_registry.build("het_sheaf_adj", self._IN, self._OUT, cfg, _REG)
        assert isinstance(model, HetSheafAdjModel)
        out = model(*self._inputs())
        assert out.shape == (self._N, self._OUT)
        assert torch.isfinite(out).all()

    def test_het_context_flags_reach_the_layers(self):
        cfg = self._cfg(ensemble=True, use_edge_types=False)
        model = model_registry.build("het_sheaf_adj", self._IN, self._OUT, cfg, _REG)
        assert isinstance(model, HetSheafAdjModel)
        generators = model.layers[0].map_generators
        assert isinstance(generators, torch.nn.ModuleList)
        assert len(generators) == self._ET
