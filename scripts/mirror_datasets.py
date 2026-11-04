#!/usr/bin/env python
"""Populate the dataset-assets release with every downloadable dataset file.

Loads each dataset once (downloading whatever is missing from its current
source), then uploads to this repository's ``dataset-assets-v1`` release any
file the release does not already have:

* the Geom-GCN split ``.npz`` files (canonical evaluation protocol);
* one ``pyg_raw__<name>.tar.gz`` per stock PyG dataset (raw subtree only --
  ``processed/`` is machine-specific and always rebuilt locally);
* the DSNN directed/synthetic data files, if present locally;
* the LINKX raw files (pokec.mat, pokec-splits.npy), if present locally.

Run it from the repo root whenever a new dataset is added:

    .venv/bin/python scripts/mirror_datasets.py

Needs a GitHub token with write access to the repo (``GITHUB_TOKEN`` env var
or the git credential helper).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from exp.datasets import assets  # noqa: E402

DATA_ROOT = os.path.join(ROOT, "exp", "data")
SPLITS_DIR = os.path.join(ROOT, "src", "exp", "splits")

# Stock PyG datasets to mirror as raw tarballs (see exp.data._PYG_RAW_MIRROR).
PYG_RAW = ["cornell", "texas", "wisconsin", "chameleon", "squirrel", "film", "telegram"]
SPLIT_BASES = [
    "cora",
    "citeseer",
    "pubmed",
    "cornell",
    "texas",
    "wisconsin",
    "chameleon",
    "squirrel",
    "film",
]


def _populate() -> None:
    """Load every dataset once so all raw files and splits are cached."""
    from exp.data import load_dataset  # noqa: PLC0415
    from exp.splits import _download_split  # noqa: PLC0415

    for base in SPLIT_BASES:
        for fold in range(10):
            path = os.path.join(SPLITS_DIR, f"{base}_split_0.6_0.2_{fold}.npz")
            if not os.path.isfile(path):
                _download_split(base, fold, path)
    for name in PYG_RAW:
        load_dataset(name, root=DATA_ROOT)


def _raw_tarball(name: str) -> bytes | None:
    """Tar the dataset's ``raw`` subtree(s) relative to the data root."""
    top = os.path.join(DATA_ROOT, name)
    raw_dirs = [
        os.path.relpath(os.path.join(cur, "raw"), DATA_ROOT)
        for cur, dirs, _ in os.walk(top)
        if "raw" in dirs
    ]
    if not raw_dirs:
        return None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in sorted(raw_dirs):
            tf.add(os.path.join(DATA_ROOT, rel), arcname=rel)
    return buf.getvalue()


def _upload(release_id: int, name: str, payload: bytes, token: str) -> None:
    req = urllib.request.Request(
        f"https://uploads.github.com/repos/{assets._REPO}/releases/"
        f"{release_id}/assets?name={name}",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=600):
        pass
    print(f"[mirror] uploaded {name} ({len(payload) / 1e6:.1f} MB)")


def main() -> int:
    _populate()

    token = assets._github_token()
    if not token:
        print("No GitHub token available (GITHUB_TOKEN or git credentials).")
        return 1
    req = urllib.request.Request(
        assets._API, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        release = json.load(resp)
    have = {a["name"] for a in release["assets"]}
    release_id = release["id"]

    n_up = 0
    for fname in sorted(os.listdir(SPLITS_DIR)):
        if fname.endswith(".npz") and fname not in have:
            with open(os.path.join(SPLITS_DIR, fname), "rb") as f:
                _upload(release_id, fname, f.read(), token)
            n_up += 1
    for name in PYG_RAW:
        asset = f"pyg_raw__{name}.tar.gz"
        if asset in have:
            continue
        payload = _raw_tarball(name)
        if payload is not None:
            _upload(release_id, asset, payload, token)
            n_up += 1
    # LINKX raw files (Google Drive originals): mirror them so loaders no
    # longer depend on Drive availability. Only mirrored if already cached.
    for fname in ("pokec.mat", "pokec-splits.npy"):
        path = os.path.join(DATA_ROOT, "pokec", "raw", fname)
        if os.path.isfile(path) and fname not in have:
            with open(path, "rb") as f:
                _upload(release_id, fname, f.read(), token)
            n_up += 1
    for sub in ("directed", "synthetic"):
        d = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.startswith(".") and fname not in have:
                with open(os.path.join(d, fname), "rb") as f:
                    _upload(release_id, fname, f.read(), token)
                n_up += 1

    print(
        f"[mirror] done, uploaded {n_up} new asset(s); release has "
        f"{len(have) + n_up} total."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
