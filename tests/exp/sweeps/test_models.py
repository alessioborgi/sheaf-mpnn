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


@pytest.mark.parametrize("model", ["nsd"])
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
