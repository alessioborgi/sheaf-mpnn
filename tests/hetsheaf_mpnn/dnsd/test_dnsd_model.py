# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


from typing import Any

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries.models import model_registry
from hetsheaf_mpnn.dnsd import HetDNSDConv, HetDNSDModel, HetDNSDVariant
from hetsheaf_mpnn.sheaf_adj import HetSheafAdjModel
from hetsheaf_mpnn.sheaf_adj.sheaf_adj_layers import BaseHetSheafAdjConv
from tests.testing import set_seed

NUM_NODES = 15
IN_CHANNELS = 10
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


@pytest.mark.parametrize("variant", list(HetDNSDVariant))
@pytest.mark.parametrize("odd", [False, True], ids=["elu", "tanh"])
@pytest.mark.parametrize("gate", [False, True], ids=["gate_off", "gate_on"])
class TestHetDNSDModel:
    @pytest.fixture
    def setup(self, variant, odd, gate):
        set_seed(42)
        model = HetDNSDModel(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=2,
            variant=variant,
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            use_node_types=True,
            use_edge_types=True,
            odd=odd,
            gate=gate,
        )
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        edge_index = _graph()
        nt, et = _types(edge_index)
        return model, x, edge_index, nt, et

    def test_model_forward_shape(self, setup):
        model, x, edge_index, nt, et = setup
        out = model(x, edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_model_gradient_flow(self, setup):
        model, x, edge_index, nt, et = setup
        model(x, edge_index, nt, et).pow(2).sum().backward()

        assert model.encoder.weight.grad is not None
        assert model.encoder.weight.grad.abs().sum() > 0
        for layer in model.layers:
            assert layer.sheaf_adj_layer.W1.grad is not None
            assert layer.sheaf_adj_layer.W1.grad.abs().sum() > 0

    def test_isolated_graph_stability(self, setup):
        model, x, _, nt, _ = setup
        model.eval()
        edge_index = torch.empty((2, 0), dtype=torch.long)
        et = torch.empty(0, dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index, nt, et)
        assert torch.isfinite(out).all()


def test_wrapped_layers_are_bare():
    model = HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, odd=True)
    for layer in model.layers:
        assert isinstance(layer, HetDNSDConv)
        assert not layer.sheaf_adj_layer.residual
        assert isinstance(layer.sheaf_adj_layer.sigma, torch.nn.Tanh)


def test_forward_without_types():
    set_seed(42)
    model = HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=2, hidden_dim=4)
    out = model(torch.randn(NUM_NODES, IN_CHANNELS), _graph())
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_node_types_reach_the_map_generator():
    set_seed(0)
    model = HetDNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=4,
        num_node_types=NUM_NODE_TYPES,
        use_node_types=True,
        variant=HetDNSDVariant.GENERAL,
    ).eval()
    # Lift the generators off their near-zero warm start so the type one-hots
    # move the maps by more than float tolerance.
    for layer in model.layers:
        assert isinstance(layer, HetDNSDConv)
        assert isinstance(layer.sheaf_adj_layer, BaseHetSheafAdjConv)
        assert isinstance(layer.sheaf_adj_layer.map_generator, torch.nn.Sequential)
        for m in layer.sheaf_adj_layer.map_generator:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = _graph()
    nt_a = torch.zeros(NUM_NODES, dtype=torch.long)
    nt_b = torch.arange(NUM_NODES) % NUM_NODE_TYPES
    with torch.no_grad():
        assert not torch.allclose(
            model(x, edge_index, nt_a), model(x, edge_index, nt_b)
        )


def test_ensemble_builds_one_generator_per_edge_type():
    model = HetDNSDModel(
        IN_CHANNELS, OUT_CHANNELS, num_edge_types=NUM_EDGE_TYPES, ensemble=True
    )
    edge_index = _graph()
    _, et = _types(edge_index)
    for layer in model.layers:
        assert isinstance(layer, HetDNSDConv)
        assert isinstance(layer.sheaf_adj_layer, BaseHetSheafAdjConv)
        generators = layer.sheaf_adj_layer.map_generators
        assert isinstance(generators, torch.nn.ModuleList)
        assert len(generators) == NUM_EDGE_TYPES
    out = model(torch.randn(NUM_NODES, IN_CHANNELS), edge_index, edge_types=et)
    assert out.shape == (NUM_NODES, OUT_CHANNELS)


def test_deltas_off_matches_het_sheaf_adj():
    # Mirrors the homogeneous invariant: with odd/gate/layer_norm off, HetDNSD
    # is numerically identical to HetSheafAdj given the same weights.
    set_seed(7)
    kwargs: dict[str, Any] = dict(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=2,
        variant=HetDNSDVariant.DIAGONAL,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
    )
    baseline = HetSheafAdjModel(**kwargs)
    dnsd = HetDNSDModel(**kwargs, layer_norm=False)
    dnsd.encoder.load_state_dict(baseline.encoder.state_dict())
    for dnsd_layer, baseline_layer in zip(dnsd.layers, baseline.layers, strict=True):
        assert isinstance(dnsd_layer, HetDNSDConv)
        assert isinstance(baseline_layer, torch.nn.Module)
        dnsd_layer.sheaf_adj_layer.load_state_dict(baseline_layer.state_dict())
    dnsd.decoder.load_state_dict(baseline.decoder.state_dict())
    dnsd.eval()
    baseline.eval()

    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = _graph()
    nt, et = _types(edge_index)
    with torch.no_grad():
        torch.testing.assert_close(
            baseline(x, edge_index, nt, et),
            dnsd(x, edge_index, nt, et),
            atol=1e-6,
            rtol=1e-6,
        )


def test_jknet_forward_and_reset():
    set_seed(42)
    model = HetDNSDModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        stalk_dim=2,
        hidden_dim=4,
        num_layers=3,
        jknet=True,
        gate=True,
        add_eps=True,
    ).eval()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    edge_index = _graph()
    before = model(x, edge_index).clone()
    assert before.shape == (NUM_NODES, OUT_CHANNELS)
    model.reset_parameters()
    assert not torch.allclose(before, model(x, edge_index))


def test_learn_alpha_false_freezes_alpha():
    model = HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, learn_alpha=False)
    for layer in model.layers:
        assert isinstance(layer, HetDNSDConv)
        assert not layer.sheaf_adj_layer.alpha.requires_grad


def test_layer_norm_false_disables_norms():
    model = HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, layer_norm=False)
    assert model.norms is None


def test_invalid_stalk_dim():
    with pytest.raises(ValueError, match="stalk_dim must be positive"):
        HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, stalk_dim=0)


def test_invalid_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, hidden_dim=0)


def test_invalid_num_layers():
    with pytest.raises(ValueError, match="must have at least one"):
        HetDNSDModel(IN_CHANNELS, OUT_CHANNELS, num_layers=0)


class TestHetDNSDRegistry:
    def _cfg(self, **kwargs) -> ModelConfig:
        defaults = dict(
            type=ModelType.HET_DNSD,
            variant="general",
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            use_node_types=True,
            use_edge_types=True,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]

    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.HET_DNSD) == "het_dnsd"
        assert "het_dnsd" in model_registry

    @pytest.mark.parametrize(
        "variant", ["diagonal", "general", "orthogonal", "low_rank"]
    )
    def test_build_returns_a_working_model(self, variant):
        edge_index = _graph()
        nt, et = _types(edge_index)
        cfg = self._cfg(variant=variant, rank=2)
        model = model_registry.build("het_dnsd", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetDNSDModel)
        out = model(torch.randn(NUM_NODES, IN_CHANNELS), edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_dnsd_flags_reach_the_model(self):
        cfg = self._cfg(odd=True, gate=True, layer_norm=True, learn_alpha=False)
        model = model_registry.build("het_dnsd", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetDNSDModel)
        assert model.norms is not None
        layer = model.layers[0]
        assert isinstance(layer, HetDNSDConv)
        assert layer.gate
        assert isinstance(layer.sheaf_adj_layer.sigma, torch.nn.Tanh)
        assert not layer.sheaf_adj_layer.alpha.requires_grad

    def test_unknown_variant_raises_with_valid_choices(self):
        with pytest.raises(ValueError, match="Invalid variant 'sheafan'"):
            model_registry.build(
                "het_dnsd",
                IN_CHANNELS,
                OUT_CHANNELS,
                self._cfg(variant="sheafan"),
                _REG,
            )
