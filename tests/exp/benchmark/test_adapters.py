from __future__ import annotations

from pathlib import Path

from exp.benchmark.adapters import build_command, hparams_path
from exp.benchmark.jobspec import JobSpec
from exp.benchmark.models import BenchmarkManifest


def _manifest(tmp_path: Path) -> BenchmarkManifest:
    ours_dir, original_dir = tmp_path / "ours", tmp_path / "original"
    ours_dir.mkdir()
    original_dir.mkdir()
    return BenchmarkManifest.model_validate(
        {
            "name": "t",
            "datasets": ["texas"],
            "variants": ["orthogonal"],
            "sides": {
                "ours": {"python": "/env/ours/bin/python", "config_dir": str(ours_dir)},
                "original": {
                    "python": "/env/original/bin/python",
                    "repo_path": "external/neural-sheaf-diffusion",
                    "config_dir": str(original_dir),
                },
            },
            "results_dir": str(tmp_path / "results"),
            "slurm": {"partition": "gpu", "account": "acct"},
        }
    )


def test_hparams_path_layout(tmp_path):
    manifest = _manifest(tmp_path)
    job = JobSpec("texas", "orthogonal", "ours", "convergence", 3)
    expected = manifest.sides.ours.config_dir / "texas" / "orthogonal.yaml"
    assert hparams_path(manifest, job) == expected


def test_build_command_ours(tmp_path):
    manifest = _manifest(tmp_path)
    job = JobSpec("texas", "orthogonal", "ours", "convergence", 3)
    argv, _cwd = build_command(manifest, job, tmp_path / "out.json")
    assert argv[0] == str(Path("/env/ours/bin/python").resolve())
    assert argv[argv.index("--dataset") + 1] == "texas"
    assert argv[argv.index("--variant") + 1] == "orthogonal"
    assert argv[argv.index("--mode") + 1] == "convergence"


def test_build_command_original_maps_variant_and_sets_cwd(tmp_path):
    manifest = _manifest(tmp_path)
    job = JobSpec("texas", "orthogonal", "original", "efficiency", 2)
    argv, cwd = build_command(manifest, job, tmp_path / "out.json")
    assert argv[0] == str(Path("/env/original/bin/python").resolve())
    assert argv[argv.index("--variant") + 1] == "BundleSheaf"
    assert cwd == Path("external/neural-sheaf-diffusion").resolve()


def test_build_command_original_passes_canonical_variant_label(tmp_path):
    # Regression: the original side used to record the original repo's class
    # name ("BundleSheaf") in its result JSON, so collect() could never pair
    # it with the "ours" side's canonical "orthogonal".
    manifest = _manifest(tmp_path)
    job = JobSpec("texas", "orthogonal", "original", "efficiency", 2)
    argv, _cwd = build_command(manifest, job, tmp_path / "out.json")
    assert argv[argv.index("--variant-label") + 1] == "orthogonal"
    assert argv[argv.index("--variant") + 1] == "BundleSheaf"


def test_build_command_uses_absolute_paths_for_original_side(tmp_path):
    # The original side runs with cwd=repo_path, so a relative interpreter
    # path fails at exec and relative --hparams/--out resolve against the
    # wrong directory.
    manifest = _manifest(tmp_path)
    job = JobSpec("texas", "orthogonal", "original", "efficiency", 2)
    argv, cwd = build_command(manifest, job, tmp_path / "out.json")
    assert Path(argv[0]).is_absolute()
    assert Path(argv[argv.index("--repo-path") + 1]).is_absolute()
    assert Path(argv[argv.index("--hparams") + 1]).is_absolute()
    assert Path(argv[argv.index("--out") + 1]).is_absolute()
    assert cwd.is_absolute()
