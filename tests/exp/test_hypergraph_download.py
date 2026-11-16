"""First-use preparation and offline cache behavior."""

from pathlib import Path

import numpy as np
import pytest

from exp.datasets import hypergraph_prepare
from exp.datasets.hypergraph import ensure_hypergraph, validate_reference_artifact


def test_first_use_then_offline_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "exp.datasets.hypergraph.validate_reference_artifact", lambda *args: None
    )
    calls = []

    def prepare(name, root, manifest, download):
        assert manifest.is_file() and download
        calls.append(name)
        path = root / f"{name}.npz"
        for file in (
            path,
            path.with_suffix(".splits.npz"),
            path.with_suffix(".provenance.json"),
        ):
            file.write_bytes(b"fixture")
        return path

    monkeypatch.setattr(hypergraph_prepare, "prepare", prepare)
    path = ensure_hypergraph("squirrel-hypergraph", str(tmp_path))
    assert ensure_hypergraph("squirrel-hypergraph", str(tmp_path)) == path
    assert calls == ["squirrel-hypergraph"]


def test_modified_split_is_rejected(tmp_path):
    source = Path("data/cora-dshn/cora-dshn-q0.npz")
    if not source.exists():
        pytest.skip("Audited local Cora cache unavailable")
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    split = source.with_suffix(".splits.npz")
    target.with_suffix(".splits.npz").write_bytes(split.read_bytes())
    validate_reference_artifact("cora-hypergraph", target)
    with np.load(split) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["train"][0, 0], arrays["test"][0, 0] = (
        arrays["test"][0, 0],
        arrays["train"][0, 0],
    )
    np.savez_compressed(target.with_suffix(".splits.npz"), **arrays)
    with pytest.raises(ValueError, match="differ from the pinned reference"):
        validate_reference_artifact("cora-hypergraph", target)
