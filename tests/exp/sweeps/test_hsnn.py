"""HSNN sweep holdout isolation, resume budgets and final-config export."""

import copy
import json
import shlex
import sys

import optuna
import pytest

from exp.registries.presets import preset_registry
from exp.sweeps import hsnn
from exp.sweeps.models import SweepConfig


def record_scores(cfg, validation):
    from pathlib import Path

    folder = Path(cfg.hypergraph.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "metric": "accuracy",
                "test_evaluated": True,
                "runs": [
                    {
                        "split_row": i,
                        "best_val_accuracy": value,
                        "test": {"accuracy": 0.1},
                    }
                    for i, value in enumerate(validation)
                ],
            }
        )
    )
    return [0.1] * len(validation)


@pytest.fixture
def study_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(hsnn, "load_provenance", lambda path: {"sha256": "data"})
    monkeypatch.setattr(hsnn, "source_hashes", lambda: {"code": "hash"})
    return SweepConfig.model_validate(
        {
            "model": "hsnn",
            "search_space": {
                "variant": {"type": "categorical", "choices": ["general"]},
                "lr": {"type": "categorical", "choices": [0.01]},
            },
            "config": {
                "storage": f"sqlite:///{(tmp_path / 'study.db').as_posix()}",
                "study_name": "unit",
                "n_trials": 2,
                "n_seeds_per_trial": 2,
                "cuda": -1,
                "seed": 0,
            },
        }
    )


def test_validation_only_budget_resume_and_export(study_spec, tmp_path, monkeypatch):
    calls = []

    def run(cfg):
        if cfg.cv.folds == 10:
            return record_scores(cfg, [0.75] * 10)
        calls.append(copy.deepcopy(cfg))
        assert cfg.hypergraph.evaluate_test
        assert cfg.cv.folds == 2 and cfg.hardware.cuda == -1
        assert cfg.model.variant == "general"
        return record_scores(cfg, [0.7, 0.8])

    monkeypatch.setattr(hsnn, "run", run)
    hsnn.sweep(study_spec, "hsnn_cora")
    hsnn.sweep(study_spec, "hsnn_cora")
    assert len(calls) == 2  # Existing finished trials consume the total budget.
    folder = tmp_path / "study-artifacts"
    result = json.loads((folder / "best.json").read_text())
    assert result["validation_mean"] == pytest.approx(0.75)
    assert result["validation_std"] == pytest.approx(0.05)
    assert result["test_evaluated_during_search"]
    study = optuna.load_study(study_name="unit", storage=study_spec.config.storage)
    assert study.best_trial.user_attrs["test_mean"] == pytest.approx(0.1)
    assert (folder / "final-trial-00000" / "summary.json").exists()
    assert result["config"]["hypergraph"]["evaluate_test"]
    assert result["config"]["cv"]["folds"] == 10
    # The exported command must parse through the real common config parser.
    from exp.run import _parse_config

    args = shlex.split((folder / "final-test.sh").read_text().splitlines()[-1])
    monkeypatch.setattr(sys, "argv", ["sheaf run", *args[4:]])
    parsed = _parse_config()
    assert parsed.model.variant == "general"
    assert parsed.hypergraph.evaluate_test and parsed.cv.folds == 10
    assert preset_registry.get("hsnn_cora").hypergraph.output is None


def test_changed_protocol_is_rejected(study_spec, monkeypatch):
    monkeypatch.setattr(
        hsnn, "run", lambda cfg: record_scores(cfg, [0.7] * cfg.cv.folds)
    )
    hsnn.sweep(study_spec, "hsnn_cora")
    study_spec.config.seed = 123
    with pytest.raises(ValueError, match="protocol/code/data changed"):
        hsnn.sweep(study_spec, "hsnn_cora")


def test_directed_sweep_uses_directed_runner_and_exports_model(
    study_spec, tmp_path, monkeypatch
):
    from exp.config import ModelType

    study_spec.model = "dhsnn"
    calls = []

    def directed(cfg):
        assert cfg.model.type == ModelType.DHSNN
        assert cfg.hypergraph.evaluate_test
        if cfg.cv.folds == 10:
            return record_scores(cfg, [0.65] * 10)
        calls.append(cfg)
        return record_scores(cfg, [0.6, 0.7])

    monkeypatch.setattr(hsnn, "run_directed", directed)
    hsnn.sweep(study_spec, "dhsnn_cora")
    hsnn.sweep(study_spec, "dhsnn_cora")
    assert len(calls) == 2
    folder = tmp_path / "study-artifacts"
    result = json.loads((folder / "best.json").read_text())
    assert result["config"]["model"]["type"] == "dhsnn"
    assert "dhsnn_cora" in (folder / "final-test.sh").read_text()
    from exp.run import _parse_config

    args = shlex.split((folder / "final-test.sh").read_text().splitlines()[-1])
    monkeypatch.setattr(sys, "argv", ["sheaf run", *args[4:]])
    parsed = _parse_config()
    assert parsed.model.dhsnn_map_activation == "sigmoid"
    assert parsed.model.dhsnn_input_norm
    assert not parsed.model.dhsnn_dynamic_sheaf


def test_cuda_oom_counts_toward_budget(study_spec, monkeypatch):
    import torch

    def oom(cfg):
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(hsnn, "run", oom)
    with pytest.raises(RuntimeError, match="No successful"):
        hsnn.sweep(study_spec, "hsnn_cora")
    study = optuna.load_study(study_name="unit", storage=study_spec.config.storage)
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)


def test_unsupported_graph_parameter_fails_before_training(study_spec):
    from exp.sweeps.models import CategoricalParam

    study_spec.search_space["alpha"] = CategoricalParam(
        type="categorical", choices=[1.0]
    )
    with pytest.raises(ValueError, match="Unsupported HSNN search"):
        hsnn.sweep(study_spec, "hsnn_cora")


def test_complete_budget_retries_runtime_failure(study_spec, monkeypatch):
    study_spec.config.budget_states = "complete"
    calls = []

    def runner(cfg):
        if cfg.cv.folds == 10:
            return record_scores(cfg, [0.75] * 10)
        calls.append(cfg)
        if len(calls) == 1:
            raise RuntimeError("synthetic numerical failure")
        return record_scores(cfg, [0.7, 0.8])

    monkeypatch.setattr(hsnn, "run", runner)
    hsnn.sweep(study_spec, "hsnn_cora")
    hsnn.sweep(study_spec, "hsnn_cora")
    assert len(calls) == 3
    study = optuna.load_study(study_name="unit", storage=study_spec.config.storage)
    assert len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) == 2


@pytest.mark.parametrize(
    "damage", ["truncated", "missing_test", "nan", "duplicate_split"]
)
def test_resume_retries_invalid_final_results(
    study_spec, tmp_path, monkeypatch, damage
):
    finals = []

    def runner(cfg):
        if cfg.cv.folds == 10:
            finals.append(cfg.hypergraph.output)
        return record_scores(cfg, [0.75] * cfg.cv.folds)

    monkeypatch.setattr(hsnn, "run", runner)
    hsnn.sweep(study_spec, "hsnn_cora")
    path = tmp_path / "study-artifacts/final-trial-00000/summary.json"
    summary = json.loads(path.read_text())
    if damage == "missing_test":
        summary["test_evaluated"] = False
    elif damage == "nan":
        summary["runs"][0]["test"]["accuracy"] = float("nan")
    elif damage == "duplicate_split":
        summary["runs"][0]["split_row"] = 1
    path.write_text("{" if damage == "truncated" else json.dumps(summary))
    hsnn.sweep(study_spec, "hsnn_cora")
    hsnn.sweep(study_spec, "hsnn_cora")
    assert len(finals) == 2
    assert finals[-1].endswith("-retry-1")
