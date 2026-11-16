"""Validate the complete hypergraph search matrix through the shared schema."""

import json
import runpy
import sys
from pathlib import Path

import yaml

from exp.datasets.hypergraph import HYPERGRAPHS
from exp.registries.presets import preset_registry
from exp.sweeps.models import SweepConfig
from exp.sweeps.sweep import _build_cfg


def test_complete_hypergraph_search_matrix():
    root = Path("configs/hypergraphs")
    paths = list(root.glob("*/*/*.yaml"))
    assert len(paths) == len(HYPERGRAPHS) * 8
    studies, databases = set(), set()
    for path in paths:
        dataset, model, variant = path.parts[-3], path.parts[-2], path.stem
        spec = SweepConfig.model_validate(yaml.safe_load(path.read_text()))
        assert spec.config.n_trials == 250
        assert spec.model == model
        assert spec.dataset.name == dataset
        assert spec.dataset.directed == (model == "dhsnn")
        assert spec.search_space["variant"].choices == [variant]
        assert spec.config.study_name not in studies
        assert spec.config.storage not in databases
        studies.add(spec.config.study_name)
        databases.add(spec.config.storage)
        cfg = _build_cfg(preset_registry.get(dataset), model, {"variant": variant})
        assert cfg.model.type.value == model
        assert cfg.model.variant == variant
        assert cfg.hypergraph.evaluate_test
        if variant == "low_rank":
            assert max(spec.search_space["rank"].choices) <= min(
                spec.search_space["stalk_dim"].choices
            )


def test_paper_search_uses_library_training_budget():
    paths = list(Path("configs/paper-dshn").glob("*/*.yaml"))
    assert len(paths) == 2 * len(HYPERGRAPHS)
    for path in paths:
        spec = SweepConfig.model_validate(yaml.safe_load(path.read_text()))
        assert spec.search_space["epochs"].choices == [1500]
        assert spec.search_space["early_stopping"].choices == [200]
        assert spec.config.n_seeds_per_trial == 10
        assert spec.config.std_weight == 0
        assert spec.config.study_name.startswith("paper1500-")
        params = {
            name: value.choices[0]
            for name, value in spec.search_space.items()
            if value.type == "categorical"
        }
        cfg = _build_cfg(preset_registry.get(spec.dataset.name), "dhsnn", params)
        assert cfg.optim.epochs == 1500
        assert cfg.model.dhsnn_prediction == "MLP_var3"


def test_limited_campaign_keeps_tagged_budget_and_log(tmp_path, monkeypatch):
    script = Path("scripts/run_hypergraph_campaign.py").resolve()
    source = Path("configs/hypergraphs/cora-hypergraph/dhsnn/diagonal.yaml").read_text()
    monkeypatch.chdir(tmp_path)
    config = Path("configs/hypergraphs/cora-hypergraph/dhsnn/diagonal.yaml")
    config.parent.mkdir(parents=True)
    config.write_text(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--dataset",
            "cora-hypergraph",
            "--model",
            "dhsnn",
            "--variant",
            "diagonal",
            "--trials",
            "3",
            "--splits",
            "1",
            "--cuda",
            "-1",
            "--tag",
            "pilot-v2",
        ],
    )
    calls = []

    def launch(command, **kwargs):
        path = Path(command[command.index("--yaml-path") + 1])
        spec = SweepConfig.model_validate(yaml.safe_load(path.read_text()))
        assert spec.config.n_trials == 3 and spec.config.n_seeds_per_trial == 1
        assert spec.config.cuda == -1
        assert spec.config.study_name.endswith("-pilot-v2")
        assert spec.config.storage.endswith("-pilot-v2.db")
        assert "pilot-v2" in kwargs["stdout"].name
        calls.append(command)

    monkeypatch.setattr("subprocess.run", launch)
    runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 1


def test_verification_reruns_after_source_change(tmp_path, monkeypatch):
    script = Path("scripts/verify_hypergraph_models.py").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("exp.datasets.hypergraph.HYPERGRAPHS", ["cora-hypergraph"])
    source = {"model.py": "old"}
    monkeypatch.setattr("exp.hsnn_run.source_hashes", lambda: source)
    calls = []

    def runner(cfg):
        calls.append(cfg)
        return [0.5]

    monkeypatch.setattr("exp.run.run", runner)
    runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 8
    runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 8
    source["model.py"] = "new"
    runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 16
    report = json.loads(Path("runs/hypergraph-verification/status.json").read_text())
    assert all(row["status"] == "passed" for row in report.values())
