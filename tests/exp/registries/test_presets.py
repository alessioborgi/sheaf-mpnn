# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the preset registry."""

import pytest

from exp.config import Config, ModelType
from exp.registries.presets import PresetRegistry, generate_config, preset_registry


class TestPresetRegistryContents:
    def test_all_14_base_datasets_registered(self):
        base = {
            "cora",
            "citeseer",
            "chameleon",
            "squirrel",
            "chameleon_filtered",
            "squirrel_filtered",
            "cornell",
            "texas",
            "film",
            "amazon_ratings",
            "minesweeper",
            "questions",
            "roman_empire",
            "tolokers",
        }
        registered = set(preset_registry.list_keys())
        assert base.issubset(registered)

    def test_total_preset_count(self):
        # 16 base NSD aliases + 16*6 NSD variant presets
        assert len(preset_registry.list_keys()) > 16

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            preset_registry.get("definitely_not_a_preset")


class TestPresetRegistryGetOrDefault:
    def test_none_returns_default_config(self):
        result = preset_registry.get_or_default(None)
        assert isinstance(result, Config)
        assert result == Config()

    def test_known_name_returns_preset(self):
        result = preset_registry.get_or_default("cora")
        assert isinstance(result, Config)
        assert result.dataset.name == "cora"

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            preset_registry.get_or_default("nonexistent_preset")


class TestPresetValues:
    def test_cora_preset_values(self):
        cfg = preset_registry.get("cora")
        assert cfg.dataset.name == "cora"
        assert cfg.model.variant == "general"
        assert cfg.model.stalk_dim == 4
        assert cfg.model.type == ModelType.NSD

    def test_texas_is_bodnar_config(self):
        cfg = preset_registry.get("texas")
        # Bodnar's run_texas.sh: BundleSheaf, sparse learner, default budget.
        assert cfg.optim.stop_strategy == "loss"
        assert cfg.optim.epochs == 1500
        assert cfg.model.variant == "orthogonal"
        assert cfg.model.sparse_learner and cfg.model.edge_weights
        assert cfg.model.orth_strategy == "householder"
        assert cfg.model.learn_alpha is False

    def test_film_uses_diagonal_variant(self):
        cfg = preset_registry.get("film")
        assert cfg.model.variant == "diagonal"

    def test_chameleon_is_bodnar_config(self):
        cfg = preset_registry.get("chameleon")
        # Bodnar's run_chameleon.sh: DiagSheaf with add_lp and second_linear.
        assert cfg.model.variant == "diagonal"
        assert cfg.model.add_lp and cfg.model.second_linear
        assert cfg.optim.stop_strategy == "acc"
        assert cfg.optim.early_stopping == 100

    def test_preset_returns_config_instance(self):
        for name in ["cora", "texas", "film", "amazon_ratings"]:
            assert isinstance(preset_registry.get(name), Config)


class TestPresetRegistryIsolation:
    """Each PresetRegistry instance is independent - useful in tests."""

    def test_fresh_registry_is_empty(self):
        r = PresetRegistry()
        assert r.list_keys() == []

    def test_fresh_registry_get_or_default_none(self):
        r = PresetRegistry()
        assert r.get_or_default(None) == Config()

    def test_registering_to_fresh_does_not_affect_global(self):
        r = PresetRegistry()
        r.register("test_preset", Config())
        assert "test_preset" not in preset_registry


class TestGenerateConfig:
    def test_generate_config_returns_config(self):
        cfg = generate_config(
            "cora",
            variant="general",
            stalk_dim=4,
            hidden_dim=32,
            num_layers=2,
            input_dropout=0.5,
            dropout=0.0,
            lr=0.01,
            weight_decay=5e-4,
        )
        assert isinstance(cfg, Config)
        assert cfg.dataset.name == "cora"
        assert cfg.model.variant == "general"
        assert cfg.model.stalk_dim == 4

    def test_generate_config_model_type_default_is_nsd(self):
        cfg = generate_config(
            "cora",
            variant="general",
            stalk_dim=4,
            hidden_dim=32,
            num_layers=2,
            input_dropout=0.0,
            dropout=0.0,
            lr=0.01,
            weight_decay=5e-4,
        )
        assert cfg.model.type == ModelType.NSD
