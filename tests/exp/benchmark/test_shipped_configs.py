# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Every YAML shipped under ``configs/benchmark`` must load with its real
loader. ``tests/exp/sweeps/test_sweep_yaml_configs.py`` deliberately excludes
this subtree (these are not sweep configs), so this is its only validation.
"""

from __future__ import annotations

from pathlib import Path

from exp.benchmark.hparams import load_original_hparams, load_ours_hparams
from exp.benchmark.models import load_manifest

_BENCHMARK_CONFIGS = Path(__file__).parents[3] / "configs" / "benchmark"


def test_example_manifest_loads():
    load_manifest(_BENCHMARK_CONFIGS / "nsd_vs_original.yaml")


def test_ours_bestconf_files_load_and_match_their_filename_variant():
    paths = sorted((_BENCHMARK_CONFIGS / "bestconf" / "ours").rglob("*.yaml"))
    assert paths, "no ours-side bestconf files found"
    for path in paths:
        ours = load_ours_hparams(path)
        # run_one rejects a best_params variant that disagrees with the job
        # variant, and the job variant is this file's stem.
        assert str(ours.best_params["variant"]) == path.stem, path


def test_original_bestconf_files_load():
    paths = sorted((_BENCHMARK_CONFIGS / "bestconf" / "original").rglob("*.yaml"))
    assert paths, "no original-side bestconf files found"
    for path in paths:
        load_original_hparams(path)
