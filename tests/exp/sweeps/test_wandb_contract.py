# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""W&B contract: every registered model must reach W&B with the same detail
set the NSD runs have - identity stamped into the run config at trial start
and a complete aggregate payload logged at trial end.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from exp.config import Config, DatasetConfig, ModelConfig, ModelType, OptimConfig
from exp.registries.models import model_registry
from exp.sweeps.models import SweepConfig
from exp.sweeps.sweep import _run_trial
from tests.exp.sweeps.test_sweep import _make_dm_mock, _make_trial

# One valid restriction-map variant per family, for families whose variants are
# not the diagonal/general/orthogonal set: attention families have no diagonal
# map, and CSNN's variants select the conformal scaling instead. Everything else
# uses ModelConfig defaults.
_FAMILY_VARIANT = {
    "sheaf_attn": "sheafan",
    "het_sheaf_attn": "sheafan",
    "csnn": "conformal",
    "het_csnn": "conformal",
}

# Fields the NSD rows carry on W&B; every model must produce the same set.
# Identity is stamped into config AND summary at trial start (wandb.log never
# runs for pruned/failed trials); wandb.log carries only the aggregates.
_STAMP_FIELDS = {"model", "dataset", "transport_type", "trial_number"}
_LOG_FIELDS = {
    "n_seeds",
    "mean_val_acc",
    "std_val_acc",
    "mean_test_acc",
    "mean_fit_time_s",
    "mean_epoch_time_s",
    "mean_epochs_trained",
}

_ALL_MODELS = sorted(model_registry.list_keys())


def _variant_for(name: str) -> str:
    return _FAMILY_VARIANT.get(name, "diagonal")


_MC_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}


def _model_cfg(name: str) -> ModelConfig:
    kwargs: dict[str, Any] = {
        "type": ModelType(name),
        "variant": cast(Any, _variant_for(name)),
    }
    # Het families embed node/edge types, which must be positive to build;
    # repos without het models do not carry these ModelConfig fields.
    for extra, value in (("num_node_types", 3), ("num_edge_types", 5)):
        if extra in _MC_FIELDS:
            kwargs[extra] = value
    return ModelConfig(**kwargs)


def test_registry_covers_every_model_type():
    """A ModelType without a registered factory could never log anything."""
    assert {m.value for m in ModelType} == set(_ALL_MODELS)


@pytest.mark.parametrize("name", _ALL_MODELS)
def test_registered_model_builds(name):
    from exp.config import RegConfig

    model = model_registry.build(name, 10, 3, _model_cfg(name), RegConfig())
    assert model is not None


@pytest.mark.parametrize("name", _ALL_MODELS)
def test_sweep_logs_full_wandb_detail_set(name):
    """Run one mocked sweep trial per model and assert the W&B payload."""
    sweep_cfg = SweepConfig.model_validate(
        {
            "model": name,
            "search_space": {"stalk_dim": {"type": "int", "low": 2, "high": 4}},
            "config": {"n_trials": 1, "seed": 42},
        }
    )
    base_cfg = Config(
        dataset=DatasetConfig(name="cora", root="/tmp/fake"),
        model=_model_cfg(name),
        optim=OptimConfig(epochs=2, early_stopping=1),
    )
    trial = _make_trial({"stalk_dim": 3})

    dm_mock = _make_dm_mock(metric="acc")
    trainer_mock = MagicMock()
    trainer_mock.current_epoch = 2
    trainer_mock.validate.return_value = [{"val_acc": 0.8}]
    trainer_mock.test.return_value = [{"test_acc": 0.75}]
    run_mock = MagicMock()
    log_mock = MagicMock()
    with (
        patch("exp.sweeps.sweep.SheafDataModule", return_value=dm_mock),
        patch("exp.sweeps.sweep.select_module_cls"),
        patch("exp.sweeps.sweep.Trainer", return_value=trainer_mock),
        patch("exp.sweeps.sweep.EarlyStopping"),
        patch("wandb.run", new=run_mock),
        patch("wandb.log", new=log_mock),
    ):
        _run_trial(trial, sweep_cfg, base_cfg)

    stamped = run_mock.config.update.call_args_list[0].args[0]
    assert set(stamped) >= _STAMP_FIELDS
    assert stamped["model"] == name
    assert stamped["dataset"] == "cora"
    assert stamped["transport_type"] == _variant_for(name)

    # Summary mirror: existing W&B tables group by the summary identity the
    # old wandb.log wrote, so the stamp must populate it too.
    summary_set = {
        c.args[0]: c.args[1] for c in run_mock.summary.__setitem__.call_args_list
    }
    assert summary_set.get("model") == name
    assert summary_set.get("dataset") == "cora"
    assert summary_set.get("transport_type") == _variant_for(name)

    logged = log_mock.call_args.args[0]
    missing = _LOG_FIELDS - set(logged)
    assert not missing, f"{name} W&B payload missing fields: {sorted(missing)}"
