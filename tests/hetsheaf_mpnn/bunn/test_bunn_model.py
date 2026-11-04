# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end HetBuNNModel: parity with BuNN, het context, registry, presets."""

import pytest
import torch

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries.models import model_registry
from exp.registries.presets import preset_registry
from hetsheaf_mpnn.bunn import HetBundleMapLearner, HetBuNNConv, HetBuNNModel
from hetsheaf_mpnn.bunn.bundle import HetSumGNNConv
from sheaf_mpnn.bunn import BuNNModel
from tests.bunn.test_bunn_bundle import ring_edge_index
from tests.testing import set_seed

NUM_NODES = 8
IN_CHANNELS = 10
OUT_CHANNELS = 3
NUM_NODE_TYPES = 2
NUM_EDGE_TYPES = 3
_REG = RegConfig()


@pytest.fixture
def edge_index() -> torch.Tensor:
    return ring_edge_index(NUM_NODES)


@pytest.fixture
def x() -> torch.Tensor:
    return torch.randn(NUM_NODES, IN_CHANNELS)


def _types(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nt = torch.arange(NUM_NODES) % NUM_NODE_TYPES
    et = torch.arange(edge_index.size(1)) % NUM_EDGE_TYPES
    return nt, et


def make_model(**kwargs) -> HetBuNNModel:
    defaults = {
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "bundle_dim": 2,
        "hidden_dim": 4,
        "num_bundles": 3,
        "num_layers": 2,
    }
    defaults.update(kwargs)
    return HetBuNNModel(**defaults)  # ty: ignore[invalid-argument-type]


def _full_het(**kwargs) -> HetBuNNModel:
    return make_model(
        phi="sum_gnn",
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        use_node_types=True,
        use_edge_types=True,
        **kwargs,
    )


@pytest.mark.parametrize("solver", ["taylor", "spectral"])
@pytest.mark.parametrize("shared_phi", [False, True])
@pytest.mark.parametrize("phi", ["mlp", "sum_gnn"])
def test_features_only_matches_bunn(solver, shared_phi, phi, x, edge_index):
    # With every het flag off, HetBuNN is BuNN: same parameters, same output.
    set_seed(0)
    kwargs = dict(solver=solver, shared_phi=shared_phi, phi=phi)
    het = make_model(**kwargs).eval()
    base = BuNNModel(
        IN_CHANNELS,
        OUT_CHANNELS,
        bundle_dim=2,
        hidden_dim=4,
        num_bundles=3,
        num_layers=2,
        **kwargs,
    ).eval()
    base.load_state_dict(het.state_dict())
    nt, et = _types(edge_index)
    torch.testing.assert_close(het(x, edge_index, nt, et), base(x, edge_index))


class TestForward:
    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    @pytest.mark.parametrize("shared_phi", [False, True])
    def test_output_shape_with_every_type(self, solver, shared_phi, x, edge_index):
        model = _full_het(solver=solver, shared_phi=shared_phi)
        nt, et = _types(edge_index)
        out = model(x, edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_forward_without_types(self, x, edge_index):
        assert _full_het()(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)

    def test_layers_and_shared_learner_are_type_aware(self):
        separate = _full_het()
        for layer in separate.layers:
            assert isinstance(layer, HetBuNNConv)
            assert isinstance(layer.map_learner, HetBundleMapLearner)
        shared = _full_het(shared_phi=True)
        assert isinstance(shared.shared_map_learner, HetBundleMapLearner)
        for layer in shared.layers:
            assert isinstance(layer, HetBuNNConv)
            assert layer.map_learner is None

    def test_node_types_change_the_output(self, x, edge_index):
        set_seed(0)
        model = make_model(num_node_types=NUM_NODE_TYPES, use_node_types=True).eval()
        zeros = torch.zeros(NUM_NODES, dtype=torch.long)
        ones = torch.ones(NUM_NODES, dtype=torch.long)
        assert not torch.allclose(
            model(x, edge_index, zeros), model(x, edge_index, ones)
        )

    def test_edge_types_start_inert_then_reach_the_output(self, x, edge_index):
        set_seed(0)
        model = _full_het().eval()
        nt, et = _types(edge_index)
        # Zero-initialised relation weights: the edge types change nothing yet.
        torch.testing.assert_close(
            model(x, edge_index, nt, et), model(x, edge_index, nt)
        )
        for module in model.modules():
            if isinstance(module, HetSumGNNConv):
                torch.nn.init.normal_(module.rel_weight)
        assert not torch.allclose(
            model(x, edge_index, nt, et), model(x, edge_index, nt)
        )

    def test_pe_and_node_types_compose(self, x, edge_index):
        model = make_model(
            pe_size=4, num_node_types=NUM_NODE_TYPES, use_node_types=True
        )
        nt, _ = _types(edge_index)
        out = model(x, edge_index, nt, pe=torch.randn(NUM_NODES, 4))
        assert out.shape == (NUM_NODES, OUT_CHANNELS)

    def test_pe_is_required_when_declared(self, x, edge_index):
        with pytest.raises(ValueError, match="pass pe to forward"):
            make_model(pe_size=4)(x, edge_index)

    def test_jknet_concatenates_layer_states(self, x, edge_index):
        model = _full_het(jknet=True, num_layers=3)
        assert model.decoder.in_features == 3 * model.width
        nt, et = _types(edge_index)
        assert model(x, edge_index, nt, et).shape == (NUM_NODES, OUT_CHANNELS)

    def test_rejects_malformed_inputs(self, edge_index):
        model = make_model()
        with pytest.raises(ValueError, match="x must have shape"):
            model(torch.randn(NUM_NODES, IN_CHANNELS + 1), edge_index)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"bundle_dim": 0}, "bundle_dim must be positive"),
            ({"num_layers": 0}, "at least one HetBuNN layer"),
            ({"phi": "transformer"}, "phi must be one of"),
            (
                {"use_edge_types": True, "num_edge_types": NUM_EDGE_TYPES},
                "requires phi='sum_gnn'",
            ),
            ({"use_features": False}, "phi has no input"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_model(**kwargs)


class TestTraining:
    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    def test_gradients_reach_every_parameter(self, solver, x, edge_index):
        model = _full_het(solver=solver, learn_tau=True)
        nt, et = _types(edge_index)
        model(x, edge_index, nt, et).sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_reset_parameters_changes_the_output(self, x, edge_index):
        model = _full_het().eval()
        nt, et = _types(edge_index)
        before = model(x, edge_index, nt, et)
        model.reset_parameters()
        assert not torch.allclose(before, model(x, edge_index, nt, et))

    def test_spectral_eigendecomposition_is_cached(self, x, edge_index):
        model = _full_het(solver="spectral").eval()
        nt, et = _types(edge_index)
        model(x, edge_index, nt, et)
        cached = model._eig_cache
        assert cached is not None
        model(x, edge_index, nt, et)
        assert model._eig_cache is cached


class TestOverSmoothing:
    """Node types are inputs phi did not produce, so a deep stack that flattens
    the features cannot flatten the frames of different types -- unlike the
    feature-only phi of ``tests/bunn``, which drifts to a trivial bundle. They
    are not injective, though: nodes of one type still collapse together.
    """

    def _out(self, model, x, edge_index, node_types=None):
        model.eval()
        model.decoder = torch.nn.Identity()
        return model(x, edge_index, node_types)

    def test_node_types_keep_types_apart_in_deep_stacks(self, x, edge_index):
        set_seed(0)
        model = make_model(
            num_layers=8,
            tau=10.0,
            solver="spectral",
            hidden_dim=8,
            num_node_types=NUM_NODE_TYPES,
            use_node_types=True,
        )
        nt, _ = _types(edge_index)
        out = self._out(model, x, edge_index, nt)
        within = torch.cdist(out[nt == 0], out[nt == 0]).max()
        between = torch.cdist(out[nt == 0], out[nt == 1]).mean()
        assert between > 1e-3
        assert within < 1e-4


class TestRegistry:
    def _cfg(self, **kwargs) -> ModelConfig:
        defaults = dict(
            type=ModelType.HET_BUNN,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            bunn_num_bundles=2,
            bunn_phi="sum_gnn",
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            use_node_types=True,
            use_edge_types=True,
        )
        defaults.update(kwargs)
        return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]

    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.HET_BUNN) == "het_bunn"
        assert "het_bunn" in model_registry

    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    def test_build_returns_a_working_model(self, solver, x, edge_index):
        cfg = self._cfg(bunn_solver=solver)
        model = model_registry.build("het_bunn", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetBuNNModel)
        assert model.width == 2 * 2 * 4
        nt, et = _types(edge_index)
        out = model(x, edge_index, nt, et)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert torch.isfinite(out).all()

    def test_het_flags_reach_the_map_learner(self):
        cfg = self._cfg(use_features=False)
        model = model_registry.build("het_bunn", IN_CHANNELS, OUT_CHANNELS, cfg, _REG)
        assert isinstance(model, HetBuNNModel)
        layer = model.layers[0]
        assert isinstance(layer, HetBuNNConv)
        learner = layer.map_learner
        assert isinstance(learner, HetBundleMapLearner)
        assert not learner.use_features
        assert learner.use_node_types and learner.use_edge_types
        assert all(isinstance(c, HetSumGNNConv) for c in learner.convs)

    def test_ensemble_is_rejected(self):
        with pytest.raises(ValueError, match="no per-edge-type ensemble"):
            model_registry.build(
                "het_bunn", IN_CHANNELS, OUT_CHANNELS, self._cfg(ensemble=True), _REG
            )

    def test_edge_types_with_an_mlp_phi_are_rejected(self):
        with pytest.raises(ValueError, match="requires phi='sum_gnn'"):
            model_registry.build(
                "het_bunn", IN_CHANNELS, OUT_CHANNELS, self._cfg(bunn_phi="mlp"), _REG
            )


@pytest.mark.parametrize("dataset", ["imdb", "dblp", "acm", "freebase"])
def test_hgb_presets_build(dataset):
    cfg = preset_registry.get(f"{dataset}_het_bunn")
    assert cfg.model.type == ModelType.HET_BUNN
    assert cfg.model.bunn_phi == "sum_gnn"
    assert cfg.model.use_node_types and cfg.model.use_edge_types
    # Presets leave the type counts at 0 for the het module to back-fill.
    cfg.model.num_node_types = NUM_NODE_TYPES
    cfg.model.num_edge_types = NUM_EDGE_TYPES
    model = model_registry.build("het_bunn", IN_CHANNELS, OUT_CHANNELS, cfg.model, _REG)
    assert isinstance(model, HetBuNNModel)
