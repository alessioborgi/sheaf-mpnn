# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
from pydantic import ValidationError

from exp.sweeps.models import (
    CategoricalParam,
    DatasetSpec,
    FloatParam,
    IntParam,
    OptunaConfig,
    SweepConfig,
)


def test_valid_float_param():
    data = {"type": "float", "low": 0.001, "high": 0.1, "log": True}
    param = FloatParam.model_validate(data)

    assert isinstance(param, FloatParam)
    assert param.type == "float"
    assert param.low == 0.001
    assert param.log is True


def test_valid_int_param_defaults():
    data = {"type": "int", "low": 2, "high": 5}
    param = IntParam.model_validate(data)

    assert isinstance(param, IntParam)
    assert param.log is False  # Default value check


def test_discriminated_union_mapping():
    raw = {
        "model": "nsd",
        "search_space": {
            "lr": {"type": "float", "low": 1e-4, "high": 1e-1},
            "num_layers": {"type": "int", "low": 2, "high": 4},
            "variant": {"type": "categorical", "choices": ["general", "diagonal"]},
        },
    }

    config = SweepConfig.model_validate(raw)

    assert isinstance(config.search_space["lr"], FloatParam)
    assert isinstance(config.search_space["num_layers"], IntParam)
    assert isinstance(config.search_space["variant"], CategoricalParam)


def test_invalid_discriminator_type():
    invalid_data = {
        "model": "nsd",
        "search_space": {"bad_param": {"type": "unknown_type", "low": 0, "high": 10}},
    }

    with pytest.raises(ValidationError) as exc_info:
        SweepConfig.model_validate(invalid_data)

    print(exc_info.value)
    assert (
        "Input tag 'unknown_type' found using 'type' does not match any of the expected tags"  # noqa: E501
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "bad_input",
    [
        {"type": "float", "low": "not-a-float", "high": 1.0},
        {"type": "int", "low": 1},
    ],
)
def test_malformed_parameter_arguments(bad_input):
    invalid_data = {"model": "nsd", "search_space": {"test_param": bad_input}}

    with pytest.raises(ValidationError):
        SweepConfig.model_validate(invalid_data)


def test_optuna_config_defaults():
    raw_data = {"model": "nsd", "search_space": {}}
    config = SweepConfig.model_validate(raw_data)

    assert isinstance(config.config, OptunaConfig)
    assert config.config.n_trials == 100  # Default fallback check
    assert config.config.study_name == "nsd-sweep"
    assert config.config.budget_states == "finished"


def test_budget_states_complete_accepted():
    raw_data = {
        "model": "nsd",
        "search_space": {},
        "config": {"budget_states": "complete"},
    }
    assert SweepConfig.model_validate(raw_data).config.budget_states == "complete"


def test_budget_states_invalid_value_raises():
    with pytest.raises(ValidationError):
        SweepConfig.model_validate(
            {
                "model": "nsd",
                "search_space": {},
                "config": {"budget_states": "everything"},
            }
        )


def test_dataset_spec_name_and_root():
    spec = DatasetSpec.model_validate({"name": "texas", "root": "my/data"})
    assert spec.name == "texas"
    assert spec.root == "my/data"


def test_dataset_spec_root_defaults_to_exp_data():
    spec = DatasetSpec.model_validate({"name": "cora"})
    assert spec.root == "exp/data"


def test_sweep_config_dataset_is_optional():
    cfg = SweepConfig.model_validate({"model": "nsd", "search_space": {}})
    assert cfg.dataset is None


def test_sweep_config_dataset_override():
    cfg = SweepConfig.model_validate(
        {
            "model": "nsd",
            "dataset": {"name": "texas"},
            "search_space": {},
        }
    )
    assert cfg.dataset is not None
    assert cfg.dataset.name == "texas"
    assert cfg.dataset.root == "exp/data"


# ---------------------------------------------------------------------------
# SweepConfig validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["nsd", "nsp", "sheaf_attn"])
def test_valid_model_types_accepted(model):
    cfg = SweepConfig.model_validate({"model": model, "search_space": {}})
    assert cfg.model == model


def test_invalid_model_type_raises():
    with pytest.raises(ValidationError, match="Unknown model"):
        SweepConfig.model_validate({"model": "gcn", "search_space": {}})


def test_valid_search_space_keys_accepted():
    cfg = SweepConfig.model_validate(
        {
            "model": "nsd",
            "search_space": {
                "stalk_dim": {"type": "int", "low": 2, "high": 8},
                "input_dropout": {"type": "float", "low": 0.0, "high": 0.8},
                "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            },
        }
    )
    assert len(cfg.search_space) == 3


def test_unknown_search_space_key_raises():
    with pytest.raises(ValidationError, match="Unknown search_space parameter"):
        SweepConfig.model_validate(
            {
                "model": "nsd",
                "search_space": {"stlak_dim": {"type": "int", "low": 2, "high": 8}},
            }
        )


def test_multiple_unknown_keys_all_reported():
    with pytest.raises(ValidationError, match="Unknown search_space parameter") as exc:
        SweepConfig.model_validate(
            {
                "model": "nsd",
                "search_space": {
                    "bad_one": {"type": "int", "low": 1, "high": 2},
                    "bad_two": {"type": "int", "low": 1, "high": 2},
                },
            }
        )
    assert "bad_one" in str(exc.value) or "bad_two" in str(exc.value)


def test_dataset_spec_directed_default_false():
    spec = DatasetSpec.model_validate({"name": "cora"})
    assert spec.directed is False


def test_dataset_spec_directed_parses():
    spec = DatasetSpec.model_validate({"name": "roman_empire", "directed": True})
    assert spec.directed is True


def test_directed_nsd_presets_keep_directed_dataset():
    from exp.config import ModelType
    from exp.registries.presets import preset_registry

    for name in ("roman_empire_directed_nsd_orthogonal", "synthetic_directed_nsd"):
        cfg = preset_registry.get(name)
        assert cfg.dataset.directed is True
        assert cfg.model.type == ModelType.DIRECTED_NSD


def test_directed_only_datasets_registered():
    from exp.registries.datasets import dataset_registry

    assert dataset_registry.get("telegram").metric == "acc"
    assert dataset_registry.get("dataset_nodes500_alpha0.1_beta0.4").metric == "acc"


def test_sweep_datamodule_honours_directed(monkeypatch):
    """The directed flag must reach SheafDataModule as to_undirected=False.

    Guards against the silent regression where presets/YAMLs set
    dataset.directed=True but the sweep trains on the symmetrised graph.
    """
    import optuna

    import exp.sweeps.sweep as sweep_mod
    from exp.registries.presets import preset_registry

    captured = {}

    class _SpyDM:
        def __init__(self, name, root="exp/data", fold=0, **kwargs):
            captured.update(kwargs, name=name)
            raise RuntimeError("spy stop")

    monkeypatch.setattr(sweep_mod, "SheafDataModule", _SpyDM)
    sweep_cfg = SweepConfig.model_validate(
        {
            "model": "directed_nsd",
            "search_space": {},
            "config": {"n_seeds_per_trial": 1},
        }
    )
    base_cfg = preset_registry.get("roman_empire_directed_nsd_orthogonal")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    try:
        sweep_mod._run_trial(trial, sweep_cfg, base_cfg)
    except Exception:
        pass  # the spy aborts the trial by design
    assert captured.get("name") == "roman_empire"
    assert captured.get("to_undirected") is False
