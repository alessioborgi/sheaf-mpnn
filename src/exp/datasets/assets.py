# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""On-demand download of dataset files hosted as GitHub release assets.

The DSNN-sourced directed/synthetic dataset files are too large to live in the
git tree, and their contents differ from the public upstream releases (e.g. the
directed ``roman_empire.npz`` has a different edge list, labels, and masks than
the yandex-research original), so they are hosted as binary assets on this
repository's ``dataset-assets-v1`` release and fetched here on first use.

Access: the release is public, so no credentials are needed.  Without a token
files are fetched from the plain ``releases/download`` URL, which is not subject
to the GitHub API rate limit.  When a token is available (``GITHUB_TOKEN`` /
``GH_TOKEN`` environment variables, set automatically in GitHub Actions, or the
local ``git credential`` helper) the authenticated API is used instead, which
also works for private forks.

The source can be redirected to a fork or mirror with the
``SHEAF_DATASET_REPO`` (``owner/name``) and ``SHEAF_DATASET_TAG`` environment
variables.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from functools import lru_cache

_REPO = os.environ.get("SHEAF_DATASET_REPO", "alessioborgi/sheaf-mpnn")
_TAG = os.environ.get("SHEAF_DATASET_TAG", "dataset-assets-v1")
_API = f"https://api.github.com/repos/{_REPO}/releases/tags/{_TAG}"
_PUBLIC = f"https://github.com/{_REPO}/releases/download/{_TAG}"


def _github_token() -> str | None:
    """Best-effort GitHub token: env vars, then the git credential helper."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    try:
        out = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("password="):
            return line.removeprefix("password=")
    return None


class _AuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when following redirects.

    GitHub redirects asset downloads to S3, which rejects requests that carry
    both the pre-signed query parameters and an Authorization header.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: PLR0913
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and new.has_header("Authorization"):
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_AuthStrippingRedirectHandler)


@lru_cache(maxsize=1)
def _asset_index() -> dict[str, str]:
    """Map asset filename -> API download URL for the dataset release."""
    headers = {"Accept": "application/vnd.github+json"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(_API, headers=headers)
    try:
        with _OPENER.open(req, timeout=30) as resp:
            release = json.load(resp)
    except Exception as exc:
        raise RuntimeError(
            f"Could not list the {_TAG!r} release of {_REPO} ({exc}). "
            "Dataset downloads need network access; check that the token in "
            "GITHUB_TOKEN / GH_TOKEN (or your git credential helper) is valid, "
            "or unset it to download anonymously."
        ) from exc
    return {a["name"]: a["url"] for a in release.get("assets", [])}


def ensure_asset(filename: str, dest: str) -> str:
    """Download release asset *filename* to *dest* unless it already exists."""
    if os.path.isfile(dest):
        return dest
    token = _github_token()
    if token:
        index = _asset_index()
        if filename not in index:
            raise FileNotFoundError(
                f"{filename!r} is not an asset of the {_TAG!r} release of {_REPO}. "
                f"Available: {sorted(index)}"
            )
        headers = {
            "Accept": "application/octet-stream",
            "Authorization": f"Bearer {token}",
        }
        url = index[filename]
    else:
        headers = {}
        url = f"{_PUBLIC}/{filename}"
    req = urllib.request.Request(url, headers=headers)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[datasets] Downloading {filename} from the {_TAG} release ...")
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    try:
        # Wrap fd first so it is closed even when the request fails: Windows
        # cannot delete a file that is still open.
        with os.fdopen(fd, "wb") as out, _OPENER.open(req, timeout=600) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp_path, dest)  # atomic: no half-written dataset files
    except BaseException as exc:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
            raise FileNotFoundError(
                f"{filename!r} was not found at {url} (HTTP 404). Check that the "
                f"{_TAG!r} release of {_REPO} is public and contains this file."
            ) from exc
        raise
    print(f"[datasets] Saved -> {dest}")
    return dest


def ensure_tarball(name: str, dest_dir: str) -> bool:
    """Download tarball asset *name* and extract it under *dest_dir*.

    Returns True on success, False when the asset does not exist or cannot be
    fetched -- callers fall back to their upstream download path in that case.
    """
    import tarfile  # noqa: PLC0415

    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, f".{name}.part-extract")
    try:
        ensure_asset(name, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest_dir, filter="data")
        return True
    except Exception:
        return False
    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)


__all__ = ["ensure_asset", "ensure_tarball"]
