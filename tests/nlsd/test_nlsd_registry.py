# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end config/registry wiring for NLSD.

These tests take the path a real run takes -- ``ModelType.NLSD`` -> registry ->
model -> optimizer -- rather than importing ``NLSDModel`` directly, so a missing
registration or an unwired config field fails here.
"""

import pytest
import torch

from exp.config import Config, ModelConfig, ModelType, OptimConfig, RegConfig
from exp.registries import model_registry, preset_registry
from sheaf_mpnn.nlsd import (
    EdgeThresholdNonlinearity,
    IdentityNonlinearity,
    MLPNonlinearity,
    NLSDModel,
    ScalarThresholdNonlinearity,
)

_REG = RegConfig()


def make_cfg(**kwargs) -> ModelConfig:
    defaults = dict(
        type=ModelType.NLSD,
        variant="orthogonal",
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]


class TestNLSDRegistration:
    def test_nlsd_registered(self):
        assert "nlsd" in model_registry

    def test_model_type_enum_value_matches_the_registry_key(self):
        assert str(ModelType.NLSD) in model_registry


_VARIANTS = ["diagonal", "orthogonal", "general", "low_rank"]


class TestBuildThroughTheRegistry:
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_build_returns_a_working_nlsd_model(self, variant):
        model = model_registry.build("nlsd", 8, 3, make_cfg(variant=variant), _REG)
        assert isinstance(model, NLSDModel)
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        out = model(x, edge_index)
        assert out.shape == (10, 3)
        assert torch.isfinite(out).all()

    def test_build_respects_architecture_fields(self):
        cfg = make_cfg(stalk_dim=3, hidden_dim=8, num_layers=3)
        model = model_registry.build("nlsd", 16, 7, cfg, _REG)
        assert isinstance(model, NLSDModel)
        assert model.stalk_dim == 3
        assert model.hidden_dim == 8
        assert len(model.layers) == 3

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
        cfg = make_cfg(nlsd_nonlinearity=name)
        model = model_registry.build("nlsd", 8, 3, cfg, _REG)
        assert isinstance(model, NLSDModel)
        assert isinstance(model.layers[0].nonlinearity, expected)

    def test_psi_field_reaches_the_model(self):
        cfg = make_cfg(nlsd_nonlinearity="scalar_threshold", nlsd_psi="cos")
        model = model_registry.build("nlsd", 8, 3, cfg, _REG)
        assert isinstance(model, NLSDModel)
        nonlin = model.layers[0].nonlinearity
        assert isinstance(nonlin, ScalarThresholdNonlinearity)
        assert nonlin.psi == "cos"

    def test_mlp_layers_field_reaches_the_model(self):
        cfg = make_cfg(nlsd_nonlinearity="mlp", nlsd_mlp_layers=3)
        model = model_registry.build("nlsd", 8, 3, cfg, _REG)
        assert isinstance(model, NLSDModel)
        nonlin = model.layers[0].nonlinearity
        assert isinstance(nonlin, MLPNonlinearity)
        linears = [m for m in nonlin.mlp if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3

    def test_init_threshold_field_reaches_the_model(self):
        cfg = make_cfg(nlsd_nonlinearity="scalar_threshold", nlsd_init_threshold=2.5)
        model = model_registry.build("nlsd", 8, 3, cfg, _REG)
        assert isinstance(model, NLSDModel)
        nonlin = model.layers[0].nonlinearity
        assert isinstance(nonlin, ScalarThresholdNonlinearity)
        torch.testing.assert_close(nonlin.threshold, torch.tensor(2.5))

    def test_rank_field_reaches_the_low_rank_model(self):
        cfg = make_cfg(variant="low_rank", stalk_dim=4, rank=3)
        model = model_registry.build("nlsd", 8, 3, cfg, _REG)
        assert isinstance(model, NLSDModel)
        assert model.layers[0].rank == 3

    def test_unsupported_variant_raises_with_valid_choices(self):
        """NLSD has no attention learner; the error should list what it has."""
        with pytest.raises(ValueError, match="Invalid variant 'general_attention'"):
            model_registry.build(
                "nlsd", 8, 3, make_cfg(variant="general_attention"), _REG
            )


class TestNLSDPresets:
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_presets_exist_for_each_variant(self, variant):
        cfg = preset_registry.get(f"cora_nlsd_{variant}")
        assert cfg.model.type == ModelType.NLSD
        assert cfg.model.variant == variant

    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_preset_builds_a_model(self, variant):
        cfg = preset_registry.get(f"texas_nlsd_{variant}")
        model = model_registry.build(str(cfg.model.type), 16, 5, cfg.model, cfg.reg)
        assert isinstance(model, NLSDModel)

    def test_no_attention_variant_presets(self):
        assert "cora_nlsd_general_attention" not in preset_registry
        assert "cora_nlsd_orthogonal_attention" not in preset_registry


class TestOptimizerParamGroups:
    """``threshold_lr`` / ``alpha_lr`` are opt-in and must not disturb others."""

    def _module(self, optim: OptimConfig, **model_kwargs):
        from exp.data import DatasetInfo
        from exp.module import SheafLightningModule

        cfg = Config(model=make_cfg(**model_kwargs), optim=optim)
        info = DatasetInfo("toy", 8, 3, 1, "acc", "pyg_mask")
        return SheafLightningModule(cfg, info)

    def test_defaults_keep_every_group_on_the_base_lr(self):
        module = self._module(
            OptimConfig(lr=0.01), nlsd_nonlinearity="scalar_threshold"
        )
        opt = module.configure_optimizers()
        assert all(group["lr"] == 0.01 for group in opt.param_groups)

    def test_threshold_lr_creates_its_own_group(self):
        module = self._module(
            OptimConfig(lr=0.01, threshold_lr=0.1),
            nlsd_nonlinearity="scalar_threshold",
        )
        opt = module.configure_optimizers()
        assert sorted({group["lr"] for group in opt.param_groups}) == [0.01, 0.1]

    def test_alpha_lr_creates_its_own_group(self):
        module = self._module(
            OptimConfig(lr=0.01, alpha_lr=0.006),
            nlsd_nonlinearity="scalar_threshold",
        )
        opt = module.configure_optimizers()
        assert sorted({group["lr"] for group in opt.param_groups}) == [0.006, 0.01]

    def test_every_parameter_lands_in_exactly_one_group(self):
        module = self._module(
            OptimConfig(lr=0.01, threshold_lr=0.1, alpha_lr=0.006),
            nlsd_nonlinearity="scalar_threshold",
        )
        opt = module.configure_optimizers()
        grouped = [p for group in opt.param_groups for p in group["params"]]
        assert len(grouped) == len(list(module.parameters()))
        assert len({id(p) for p in grouped}) == len(grouped)

    def test_other_families_are_unaffected(self):
        """NSD exposes neither parameter, so the groups stay as they were."""
        module = self._module(
            OptimConfig(lr=0.01, threshold_lr=0.1, alpha_lr=0.006),
            type=ModelType.NSD,
            variant="general",
        )
        opt = module.configure_optimizers()
        assert all(group["lr"] == 0.01 for group in opt.param_groups)
