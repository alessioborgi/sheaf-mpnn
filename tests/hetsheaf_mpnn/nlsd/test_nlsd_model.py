# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end tests for the heterogeneous NLSD model and its registry wiring."""

import pytest
import torch
from torch_geometric.utils import erdos_renyi_graph

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries import model_registry
from hetsheaf_mpnn.nlsd import HetNLSDModel, HetNLSDVariant
from sheaf_mpnn.nlsd import (
    EdgeThresholdNonlinearity,
    IdentityNonlinearity,
    MLPNonlinearity,
    NLSDNonlinearity,
    ScalarThresholdNonlinearity,
)

NUM_NODES = 12
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4

_REG = RegConfig()


def _graph():
    torch.manual_seed(42)
    return erdos_renyi_graph(NUM_NODES, edge_prob=0.4)


def _types(edge_index):
    nt = torch.randint(0, NUM_NODE_TYPES, (NUM_NODES,))
    et = torch.randint(0, NUM_EDGE_TYPES, (edge_index.size(1),))
    return nt, et


def make_model(**kwargs) -> HetNLSDModel:
    defaults = dict(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        stalk_dim=4,
        hidden_dim=8,
        num_layers=2,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
    )
    defaults.update(kwargs)
    return HetNLSDModel(**defaults)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("variant", list(HetNLSDVariant))
@pytest.mark.parametrize("nonlinearity", list(NLSDNonlinearity))
class TestHetNLSDModel:
    def test_construction_and_forward(self, variant, nonlinearity):
        edge_index = _graph()
        nt, et = _types(edge_index)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        model = make_model(variant=variant, nonlinearity=nonlinearity, rank=2)
        out = model(x, edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, variant, nonlinearity):
        edge_index = _graph()
        nt, et = _types(edge_index)
        x = torch.randn(NUM_NODES, IN_CHANNELS)
        model = make_model(variant=variant, nonlinearity=nonlinearity, rank=2)
        model(x, edge_index, nt, et).pow(2).sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_forward_without_types():
    """Model works without type tensors (features-only mode)."""
    edge_index = _graph()
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = HetNLSDModel(
        IN_CHANNELS, OUT_CHANNELS, use_features=True, use_node_types=False
    )
    assert model(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)


def test_jknet_widens_the_decoder():
    model = make_model(num_layers=3, jknet=True)
    assert model.decoder.in_features == 3 * model.stalk_dim * model.hidden_dim


def test_ensemble_builds_one_generator_per_edge_type():
    edge_index = _graph()
    nt, et = _types(edge_index)
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = make_model(ensemble=True, use_edge_types=False)
    generators = model.layers[0].map_generators
    assert isinstance(generators, torch.nn.ModuleList)
    assert len(generators) == NUM_EDGE_TYPES
    assert model(x, edge_index, nt, et).shape == (NUM_NODES, OUT_CHANNELS)


class TestHetNLSDModelWiring:
    @pytest.mark.parametrize(
        "member,expected",
        [
            (NLSDNonlinearity.MLP, MLPNonlinearity),
            (NLSDNonlinearity.SCALAR_THRESHOLD, ScalarThresholdNonlinearity),
            (NLSDNonlinearity.EDGE_THRESHOLD, EdgeThresholdNonlinearity),
            (NLSDNonlinearity.IDENTITY, IdentityNonlinearity),
        ],
    )
    def test_nonlinearity_class(self, member, expected):
        model = make_model(nonlinearity=member)
        assert isinstance(model.layers[0].nonlinearity, expected)

    def test_each_layer_owns_its_own_phi(self):
        """Phi is learned per diffusion step, like the restriction maps."""
        model = make_model(num_layers=3, nonlinearity=NLSDNonlinearity.MLP)
        phis = {id(layer.nonlinearity) for layer in model.layers}
        assert len(phis) == 3

    def test_per_stalk_alpha_defaults_follow_the_reference(self):
        threshold = make_model(nonlinearity=NLSDNonlinearity.SCALAR_THRESHOLD)
        mlp = make_model(nonlinearity=NLSDNonlinearity.MLP)
        assert threshold.layers[0].per_stalk_alpha
        assert not mlp.layers[0].per_stalk_alpha

    def test_learn_alpha_false_freezes_step_size(self):
        model = make_model(learn_alpha=False)
        assert not model.layers[0].alpha.requires_grad

    def test_rank_reaches_the_low_rank_layers(self):
        model = make_model(variant=HetNLSDVariant.LOW_RANK, rank=3)
        assert model.layers[0].rank == 3

    def test_no_attention_variants(self):
        """NLSD has no attention learner, in the het setting either."""
        names = {v.name for v in HetNLSDVariant}
        assert not any("ATTENTION" in n for n in names)


class TestHetNLSDModelValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stalk_dim": 0},
            {"hidden_dim": 0},
            {"num_layers": 0},
            {"psi": "quadratic"},
        ],
    )
    def test_invalid_construction_raises(self, kwargs):
        with pytest.raises(ValueError):
            make_model(**kwargs)

    def test_bad_feature_width_raises(self):
        model = make_model()
        with pytest.raises(ValueError, match="x must have shape"):
            model(torch.randn(NUM_NODES, IN_CHANNELS + 1), _graph())

    def test_bad_edge_index_shape_raises(self):
        model = make_model()
        with pytest.raises(ValueError, match="edge_index must have shape"):
            model(
                torch.randn(NUM_NODES, IN_CHANNELS),
                torch.zeros(3, 5, dtype=torch.long),
            )


class TestHetNLSDRegistry:
    def _cfg(self, **kwargs) -> ModelConfig:
        defaults = dict(
            type=ModelType.HET_NLSD,
            variant="orthogonal",
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

    def test_het_nlsd_registered(self):
        assert "het_nlsd" in model_registry

    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.HET_NLSD) in model_registry

    @pytest.mark.parametrize(
        "variant", ["diagonal", "orthogonal", "general", "low_rank"]
    )
    def test_build_returns_a_working_model(self, variant):
        edge_index = _graph()
        nt, et = _types(edge_index)
        cfg = self._cfg(variant=variant, rank=2)
        model = model_registry.build("het_nlsd", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetNLSDModel)
        out = model(torch.randn(NUM_NODES, IN_CHANNELS), edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("mlp", MLPNonlinearity),
            ("scalar_threshold", ScalarThresholdNonlinearity),
            ("edge_threshold", EdgeThresholdNonlinearity),
            ("identity", IdentityNonlinearity),
        ],
    )
    def test_nonlinearity_field_selects_phi(self, name, expected):
        cfg = self._cfg(nlsd_nonlinearity=name)
        model = model_registry.build("het_nlsd", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetNLSDModel)
        assert isinstance(model.layers[0].nonlinearity, expected)

    def test_het_context_flags_reach_the_layers(self):
        cfg = self._cfg(ensemble=True, use_edge_types=False)
        model = model_registry.build("het_nlsd", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetNLSDModel)
        generators = model.layers[0].map_generators
        assert isinstance(generators, torch.nn.ModuleList)
        assert len(generators) == NUM_EDGE_TYPES

    def test_unsupported_variant_raises_with_valid_choices(self):
        with pytest.raises(ValueError, match="Invalid variant 'general_attention'"):
            model_registry.build(
                "het_nlsd",
                IN_CHANNELS,
                OUT_CHANNELS,
                self._cfg(variant="general_attention"),
                _REG,
            )
