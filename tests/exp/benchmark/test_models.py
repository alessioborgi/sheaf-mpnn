from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from exp.benchmark.models import BenchmarkManifest, RepeatsConfig, load_manifest


def _base_manifest_dict(tmp_path: Path) -> dict:
    ours_dir = tmp_path / "ours"
    original_dir = tmp_path / "original"
    ours_dir.mkdir()
    original_dir.mkdir()
    return {
        "name": "test",
        "datasets": ["texas"],
        "variants": ["orthogonal"],
        "sides": {
            "ours": {"python": ".venv/bin/python", "config_dir": str(ours_dir)},
            "original": {
                "python": "external/neural-sheaf-diffusion/.venv/bin/python",
                "repo_path": "external/neural-sheaf-diffusion",
                "config_dir": str(original_dir),
            },
        },
        "results_dir": str(tmp_path / "results"),
        "slurm": {"partition": "gpu", "account": "acct"},
    }


def test_manifest_parses_with_defaults(tmp_path):
    manifest = BenchmarkManifest.model_validate(_base_manifest_dict(tmp_path))
    assert manifest.repeats.folds == 10
    assert manifest.modes == ["convergence", "efficiency"]
    assert manifest.sides.original.variant_map["orthogonal"] == "BundleSheaf"


def test_manifest_rejects_unknown_variant(tmp_path):
    data = _base_manifest_dict(tmp_path)
    data["variants"] = ["low_rank"]
    with pytest.raises(ValueError, match="Unknown variant"):
        BenchmarkManifest.model_validate(data)


def test_manifest_rejects_incomplete_variant_map(tmp_path):
    data = _base_manifest_dict(tmp_path)
    data["variants"] = ["diagonal", "orthogonal"]
    data["sides"]["original"]["variant_map"] = {"diagonal": "DiagSheaf"}
    with pytest.raises(ValueError, match="variant_map is missing"):
        BenchmarkManifest.model_validate(data)


def test_repeats_rejects_warmup_at_or_above_cap():
    with pytest.raises(ValueError, match="warmup_epochs"):
        RepeatsConfig(warmup_epochs=30, efficiency_epoch_cap=30)


def test_manifest_resolves_relative_paths_to_absolute(tmp_path):
    # Regression: original-side jobs run with cwd=repo_path, so relative
    # manifest paths (the shipped example manifest's are all relative) would
    # fail at exec time or write results into the vendored submodule.
    data = _base_manifest_dict(tmp_path)
    data["sides"]["ours"]["config_dir"] = "configs/benchmark/bestconf/ours"
    data["sides"]["original"]["config_dir"] = "configs/benchmark/bestconf/original"
    data["results_dir"] = "results/benchmark/nsd_vs_original"
    manifest = BenchmarkManifest.model_validate(data)

    assert Path(manifest.sides.ours.python).is_absolute()
    assert manifest.sides.ours.config_dir.is_absolute()
    assert Path(manifest.sides.original.python).is_absolute()
    assert manifest.sides.original.repo_path.is_absolute()
    assert manifest.sides.original.config_dir.is_absolute()
    assert manifest.results_dir.is_absolute()
    assert manifest.results_dir == (Path.cwd() / "results/benchmark/nsd_vs_original")


def test_manifest_leaves_absolute_paths_unchanged(tmp_path):
    data = _base_manifest_dict(tmp_path)
    manifest = BenchmarkManifest.model_validate(data)
    assert manifest.sides.ours.config_dir == tmp_path / "ours"
    assert manifest.results_dir == tmp_path / "results"


def test_load_manifest_from_yaml(tmp_path):
    data = _base_manifest_dict(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.dump(data))
    manifest = load_manifest(manifest_path)
    assert manifest.name == "test"
