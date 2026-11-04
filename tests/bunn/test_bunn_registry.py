# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end config/registry wiring for BuNN.

These take the path a real run takes -- ``ModelType.BUNN`` -> registry ->
model -> optimizer -- rather than importing ``BuNNModel`` directly, so a
missing registration or an unwired config field fails here.
"""

import pytest
import torch

from exp.config import Config, ModelConfig, ModelType, OptimConfig, RegConfig
from exp.registries import model_registry, preset_registry
from sheaf_mpnn.bunn import BuNNConv, BuNNModel
from tests.bunn.test_bunn_bundle import ring_edge_index

_REG = RegConfig()
NUM_NODES = 8


def make_cfg(**kwargs) -> ModelConfig:
    defaults = dict(
        type=ModelType.BUNN,
        variant="orthogonal",
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        bunn_num_bundles=3,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]


class TestRegistration:
    def test_bunn_registered(self):
        assert "bunn" in model_registry

    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.BUNN) in model_registry


class TestBuildThroughTheRegistry:
    def test_build_returns_a_working_model(self):
        model = model_registry.build("bunn", 8, 3, make_cfg(), _REG)
        assert isinstance(model, BuNNModel)

        out = model(torch.randn(NUM_NODES, 8), ring_edge_index(NUM_NODES))
        assert out.shape == (NUM_NODES, 3)
        assert torch.isfinite(out).all()

    def test_architecture_fields_reach_the_model(self):
        cfg = make_cfg(stalk_dim=4, hidden_dim=8, num_layers=3, bunn_num_bundles=2)
        model = model_registry.build("bunn", 16, 7, cfg, _REG)
        assert isinstance(model, BuNNModel)

        assert model.bundle_dim == 4
        assert model.hidden_dim == 8
        assert model.num_bundles == 2
        assert len(model.layers) == 3
        assert model.width == 2 * 4 * 8

    @pytest.mark.parametrize("solver", ["taylor", "spectral"])
    def test_solver_field_selects_the_algorithm(self, solver):
        model = model_registry.build("bunn", 8, 3, make_cfg(bunn_solver=solver), _REG)
        assert isinstance(model, BuNNModel)
        assert model.solver == solver
        assert model.layers[0].solver == solver

    def test_diffusion_time_fields_reach_the_layers(self):
        cfg = make_cfg(bunn_tau=10.0, bunn_learn_tau=False, bunn_solver="spectral")
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        layer = model.layers[0]
        assert isinstance(layer, BuNNConv)
        assert torch.allclose(layer.tau, torch.full((3,), 10.0))

    def test_learn_tau_field_makes_time_a_parameter(self):
        cfg = make_cfg(bunn_learn_tau=True)
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        assert isinstance(model.layers[0].tau, torch.nn.Parameter)

    def test_max_degree_field_reaches_the_layers(self):
        model = model_registry.build("bunn", 8, 3, make_cfg(bunn_max_degree=16), _REG)
        assert isinstance(model, BuNNModel)
        assert model.layers[0].max_degree == 16

    @pytest.mark.parametrize("phi", ["mlp", "sum_gnn"])
    def test_phi_field_selects_the_map_learner(self, phi):
        model = model_registry.build("bunn", 8, 3, make_cfg(bunn_phi=phi), _REG)
        assert isinstance(model, BuNNModel)
        layer = model.layers[0]
        assert isinstance(layer, BuNNConv)
        assert layer.map_learner is not None
        assert layer.map_learner.phi == phi

    def test_shared_phi_field_moves_the_learner_to_the_model(self):
        cfg = make_cfg(bunn_shared_phi=True, num_layers=2)
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        assert model.shared_map_learner is not None
        assert all(layer.map_learner is None for layer in model.layers)

    def test_normalize_field_reaches_the_layers(self):
        cfg = make_cfg(bunn_normalize="rw")
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        assert model.layers[0].normalize == "rw"

    def test_spectral_with_random_walk_is_rejected(self):
        cfg = make_cfg(bunn_solver="spectral", bunn_normalize="rw")
        with pytest.raises(ValueError, match="spectral solver requires"):
            model_registry.build("bunn", 8, 3, cfg, _REG)

    def test_num_eigs_field_reaches_the_layers(self):
        cfg = make_cfg(bunn_solver="spectral", bunn_num_eigs=4)
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        assert model.num_eigs == 4

        out = model(torch.randn(NUM_NODES, 8), ring_edge_index(NUM_NODES))
        assert out.shape == (NUM_NODES, 3)

    def test_num_eigs_zero_means_the_exact_full_spectrum(self):
        cfg = make_cfg(bunn_solver="spectral", bunn_num_eigs=0)
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        assert model.num_eigs is None

    def test_an_unusable_taylor_time_is_refused_at_build_time(self):
        """A sweep that pairs the paper's t=100 with the default solver must
        fail loudly here rather than train on a diverged kernel.
        """
        cfg = make_cfg(bunn_solver="taylor", bunn_tau=100.0)
        with pytest.raises(ValueError, match="Taylor solver can evaluate"):
            model_registry.build("bunn", 8, 3, cfg, _REG)

    def test_the_same_time_is_fine_on_the_spectral_solver(self):
        cfg = make_cfg(bunn_solver="spectral", bunn_tau=100.0)
        model = model_registry.build("bunn", 8, 3, cfg, _REG)
        assert isinstance(model, BuNNModel)
        out = model(torch.randn(NUM_NODES, 8), ring_edge_index(NUM_NODES))
        assert torch.isfinite(out).all()

    def test_pe_size_field_reaches_the_model(self):
        model = model_registry.build("bunn", 8, 3, make_cfg(pe_size=6), _REG)
        assert isinstance(model, BuNNModel)
        assert model.pe_size == 6

        out = model(
            torch.randn(NUM_NODES, 8),
            ring_edge_index(NUM_NODES),
            torch.randn(NUM_NODES, 6),
        )
        assert out.shape == (NUM_NODES, 3)

    def test_regularisation_fields_reach_the_model(self):
        reg = RegConfig(input_dropout=0.3, dropout=0.4)
        model = model_registry.build("bunn", 8, 3, make_cfg(), reg)
        assert isinstance(model, BuNNModel)
        assert model.input_dropout_layer.p == 0.3
        assert model.dropout_layer.p == 0.4


class TestPresets:
    def test_every_node_dataset_has_a_bunn_preset(self):
        cora = preset_registry.get("cora_bunn")
        assert cora.model.type is ModelType.BUNN

    def test_preset_width_matches_the_sibling_families(self):
        """The family grid only means anything if the rows are comparable.

        BuNN's state is ``num_bundles * stalk_dim * hidden_dim`` wide, so an
        unpinned bundle count would silently give it several times the width
        of every other family at the same shared per-dataset row.
        """
        for dataset in ("cora", "texas", "minesweeper"):
            bunn = preset_registry.get(f"{dataset}_bunn").model
            nsd = preset_registry.get(f"{dataset}_nsd_general").model
            bunn_width = bunn.bunn_num_bundles * bunn.stalk_dim * bunn.hidden_dim
            assert bunn_width == nsd.stalk_dim * nsd.hidden_dim

    def test_presets_use_a_solver_their_diffusion_time_supports(self):
        for name in preset_registry.list_keys():
            if not name.endswith("_bunn"):
                continue
            cfg = preset_registry.get(name).model
            model_registry.build("bunn", 8, 3, cfg, RegConfig())

    def test_preset_builds(self):
        cfg = preset_registry.get("cora_bunn")
        model = model_registry.build("bunn", 8, 3, cfg.model, cfg.reg)
        assert isinstance(model, BuNNModel)
        out = model(torch.randn(NUM_NODES, 8), ring_edge_index(NUM_NODES))
        assert out.shape == (NUM_NODES, 3)


class TestOptimizerParamGroups:
    """``sheaf_decay`` must reach BuNN's per-node phi.

    It is the flat-bundle analogue of the sheaf families' per-edge map
    generator, so the reference's separate weight decay for map learners
    applies to it just the same.
    """

    def _module(self, optim: OptimConfig, **model_kwargs):
        from exp.data import DatasetInfo
        from exp.module import SheafLightningModule

        cfg = Config(model=make_cfg(**model_kwargs), optim=optim)
        info = DatasetInfo("toy", 8, 3, 1, "acc", "pyg_mask")
        return SheafLightningModule(cfg, info)

    def test_sheaf_decay_reaches_the_bundle_learner(self):
        module = self._module(OptimConfig(lr=0.01, weight_decay=5e-4, sheaf_decay=0.1))
        opt = module.configure_optimizers()

        learners = {id(p) for n, p in module.named_parameters() if "map_learner" in n}
        assert learners

        decayed = {
            id(p)
            for group in opt.param_groups
            if group["weight_decay"] == 0.1
            for p in group["params"]
        }
        assert learners <= decayed

    def test_shared_phi_learner_also_gets_the_sheaf_decay(self):
        module = self._module(
            OptimConfig(lr=0.01, weight_decay=5e-4, sheaf_decay=0.1),
            bunn_shared_phi=True,
            num_layers=2,
        )
        opt = module.configure_optimizers()

        learners = {
            id(p) for n, p in module.named_parameters() if "shared_map_learner" in n
        }
        assert learners

        decayed = {
            id(p)
            for group in opt.param_groups
            if group["weight_decay"] == 0.1
            for p in group["params"]
        }
        assert learners <= decayed

    def test_every_parameter_lands_in_exactly_one_group(self):
        module = self._module(OptimConfig(lr=0.01, sheaf_decay=0.1))
        opt = module.configure_optimizers()

        grouped = [id(p) for group in opt.param_groups for p in group["params"]]
        assert sorted(grouped) == sorted(id(p) for p in module.parameters())

    def test_defaults_keep_every_group_on_the_base_lr(self):
        module = self._module(OptimConfig(lr=0.01))
        opt = module.configure_optimizers()
        assert all(group["lr"] == 0.01 for group in opt.param_groups)
