# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Integration tests for the shipped sweep definitions under ``configs/``.

Every shipped YAML silently depends on the Config fields its search space
routes into: a swept name that no longer reaches a live config knob would be
suggested by Optuna and then quietly ignored. Building a model at both extremes
of each search space pins that wiring.

The cheap structural checks (schema, documented preset, directory agreement)
run in the default suite -- see ``tests/exp/sweeps/test_sweep_yaml_configs.py``.
"""

import re
from pathlib import Path

import pytest
import torch
import yaml

from exp.registries.models import model_registry
from exp.registries.presets import preset_registry
from exp.sweeps.models import CategoricalParam, FloatParam, IntParam, SweepConfig
from exp.sweeps.sweep import _build_cfg, sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
# configs/benchmark holds efficiency-benchmark manifests, not sweep configs.
CONFIG_FILES = sorted(
    p
    for p in CONFIGS_DIR.rglob("*.yaml")
    if "benchmark" not in p.relative_to(CONFIGS_DIR).parts
)


def _rel_id(path: Path) -> str:
    return str(path.relative_to(CONFIGS_DIR))


def _documented_preset(path: Path) -> str:
    """Preset key from the 'Run with' header, the documented invocation."""
    match = re.search(r"--preset\s+(\S+)", path.read_text())
    assert match, f"{_rel_id(path)} documents no --preset in its header"
    return match.group(1)


def _corner(
    spec: FloatParam | IntParam | CategoricalParam, low: bool
) -> float | int | str | list[int]:
    """Deterministic extreme of a search-space parameter."""
    if isinstance(spec, CategoricalParam):
        return spec.choices[0] if low else spec.choices[-1]
    return spec.low if low else spec.high


@pytest.mark.integration
def test_configs_dir_is_populated():
    """Guard the rglob: an empty list would silently skip every test below."""
    assert CONFIG_FILES, f"no sweep YAMLs found under {CONFIGS_DIR}"


def _supply_het_type_counts(cfg) -> None:
    """Give het models the type counts a dataset would supply.

    ``num_node_types`` / ``num_edge_types`` describe the loaded graph, not the
    search space, so het sweep YAMLs do not sweep them: at run time
    ``HetSheafLightningModule`` back-fills both from the dataset's
    ``HetDatasetInfo`` before the model is built. This test builds without a
    dataset, so it stands in for that back-fill -- the het context validator
    rejects an enabled ``use_*_types`` flag with a zero count, which is exactly
    the state the corners would otherwise hit.
    """
    if "HET_" not in str(cfg.model.type).upper():
        return
    if cfg.model.num_node_types == 0:
        cfg.model.num_node_types = 3
    if cfg.model.num_edge_types == 0:
        cfg.model.num_edge_types = 4


@pytest.mark.integration
@pytest.mark.parametrize("path", CONFIG_FILES, ids=_rel_id)
def test_shipped_sweep_config_builds_at_search_space_corners(path):
    """Each shipped sweep YAML must yield a buildable model at both extremes
    of its search space, so every swept field still routes into a live knob.
    """
    sweep_cfg = SweepConfig.model_validate(yaml.safe_load(path.read_text()))
    base = preset_registry.get(_documented_preset(path))

    for low in (True, False):
        params = {
            name: _corner(spec, low) for name, spec in sweep_cfg.search_space.items()
        }
        cfg = _build_cfg(base, sweep_cfg.model, params)
        _supply_het_type_counts(cfg)
        model = model_registry.build(sweep_cfg.model, 10, 3, cfg.model, cfg.reg)
        assert model is not None


@pytest.mark.integration
def test_shipped_config_sweep_e2e(tmp_path, monkeypatch):
    """The documented invocation for one shipped config runs end-to-end on a
    tiny budget (1 trial, 1 seed, 1 epoch) and writes its bestconf artifact.
    """
    src = CONFIGS_DIR / "texas" / "nsd" / "diagonal.yaml"
    raw = yaml.safe_load(src.read_text())
    # Shrink the sweep-scale knobs (trials, seeds, storage, wandb) and pin the
    # training budget through the search space; everything else stays shipped.
    raw["config"] = {
        "n_trials": 1,
        "n_seeds_per_trial": 1,
        "study_name": "shipped-config-e2e",
        "seed": 42,
    }
    raw["dataset"] = {"name": "texas", "root": str(REPO_ROOT / "exp" / "data")}
    raw["search_space"]["epochs"] = {"type": "int", "low": 1, "high": 1}
    raw["search_space"]["early_stopping"] = {"type": "int", "low": 2, "high": 2}

    yaml_path = tmp_path / src.name
    yaml_path.write_text(yaml.safe_dump(raw))

    # Force the CPU path: sweep() auto-selects GPU whenever CUDA is available,
    # and a busy device would fail this otherwise CPU-cheap test.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    sweep(yaml_path, preset=_documented_preset(src))

    saved = yaml.safe_load((tmp_path / "texas_nsd_bestconf.yaml").read_text())
    assert saved["model"] == "nsd"
    assert set(saved["best_params"]) == set(raw["search_space"])
