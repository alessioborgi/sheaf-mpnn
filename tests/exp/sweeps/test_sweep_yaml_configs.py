# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Every sweep YAML under ``configs/`` must load, and its preset must exist.

``scripts/sweep_platonov_family.sh`` greps the ``--preset`` line from each
YAML header as the only source of a sweep's dataset+variant identity, so a
YAML that fails schema validation, documents no preset, or names an
unregistered preset would either be skipped or run with the wrong identity.

These are the cheap structural checks, run in the default suite. The costly
counterpart -- building a model at both extremes of every search space -- lives
in ``tests/integration/test_sweep_configs.py``.
"""

import re
from pathlib import Path

import pytest
import yaml

from exp.registries.presets import preset_registry
from exp.sweeps.models import SweepConfig
from exp.sweeps.sweep import _KNOWN_VARIANTS

_CONFIGS_DIR = Path(__file__).parents[3] / "configs"
# configs/benchmark holds efficiency-benchmark manifests and bestconf
# hyperparameter files, not sweep configs; they load through
# exp.benchmark's own models and are covered by
# tests/exp/benchmark/test_shipped_configs.py.
_YAML_PATHS = sorted(
    p
    for p in _CONFIGS_DIR.rglob("*.yaml")
    if "benchmark" not in p.relative_to(_CONFIGS_DIR).parts
)


def _rel(path: Path) -> str:
    return str(path.relative_to(_CONFIGS_DIR))


def test_configs_found() -> None:
    """Guards against the glob silently matching nothing after a move."""
    assert len(_YAML_PATHS) > 200


@pytest.mark.parametrize("path", _YAML_PATHS, ids=_rel)
def test_yaml_loads_and_preset_resolves(path: Path) -> None:
    text = path.read_text()
    # Same load path as exp.sweeps.sweep: safe_load then SweepConfig.
    sweep_cfg = SweepConfig.model_validate(yaml.safe_load(text))
    match = re.search(r"--preset ([a-z_]+)", text)
    assert match is not None, "YAML header must document its --preset"
    base = preset_registry.get(match.group(1))

    # configs/<dataset>/<family>/<variant>.yaml: the path is the sweep's
    # identity, and only the preset actually sets dataset and variant. A
    # mismatch silently trains -- and logs to W&B -- something other than what
    # the location and study name claim.
    dataset_dir, family_dir = path.relative_to(_CONFIGS_DIR).parts[:2]
    assert sweep_cfg.model == family_dir, (
        f"{_rel(path)}: YAML sets model={sweep_cfg.model!r}, expected "
        f"{family_dir!r} from its directory"
    )
    # A YAML that pins its own dataset: block is the authority, and its
    # directory is then a grouping label (configs/synthetic/ holds one
    # directory of generated DSBM graphs), so only check the preset-driven case.
    if sweep_cfg.dataset is None:
        assert base.dataset.name == dataset_dir, (
            f"{_rel(path)}: preset {match.group(1)!r} is for dataset "
            f"{base.dataset.name!r}, expected {dataset_dir!r}"
        )
    # Only for the variant-grid families: families with a single shipped config
    # (conn_nsd, id_sheaf, ...) name the file after the family, not a variant.
    # Same rule sweep() applies at runtime before warning about a mismatch.
    if path.stem in _KNOWN_VARIANTS and "variant" not in sweep_cfg.search_space:
        assert base.model.variant == path.stem, (
            f"{_rel(path)}: preset {match.group(1)!r} sets "
            f"variant={base.model.variant!r}, expected {path.stem!r}"
        )
