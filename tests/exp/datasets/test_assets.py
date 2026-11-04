# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import io
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest

from exp.datasets import assets


def test_anonymous_download_uses_public_release_url(tmp_path):
    dest = tmp_path / "sub" / "x.npz"
    with (
        patch.object(assets, "_github_token", return_value=None),
        patch.object(assets, "_asset_index") as index,
        patch.object(
            assets._OPENER, "open", return_value=io.BytesIO(b"payload")
        ) as opener,
    ):
        assert assets.ensure_asset("x.npz", str(dest)) == str(dest)
    index.assert_not_called()  # no rate-limited API call without a token
    req = opener.call_args.args[0]
    assert req.full_url == f"{assets._PUBLIC}/x.npz"
    assert not req.has_header("Authorization")
    assert dest.read_bytes() == b"payload"


def test_anonymous_404_raises_file_not_found_and_cleans_up(tmp_path):
    dest = tmp_path / "x.npz"
    err = urllib.error.HTTPError("u", 404, "Not Found", Message(), None)
    with (
        patch.object(assets, "_github_token", return_value=None),
        patch.object(assets._OPENER, "open", side_effect=err),
        pytest.raises(FileNotFoundError, match="HTTP 404"),
    ):
        assets.ensure_asset("x.npz", str(dest))
    assert list(tmp_path.iterdir()) == []  # no partial file left behind


def test_token_download_uses_api_asset_url(tmp_path):
    dest = tmp_path / "x.npz"
    with (
        patch.object(assets, "_github_token", return_value="tok"),
        patch.object(assets, "_asset_index", return_value={"x.npz": "https://api/1"}),
        patch.object(
            assets._OPENER, "open", return_value=io.BytesIO(b"payload")
        ) as opener,
    ):
        assets.ensure_asset("x.npz", str(dest))
    req = opener.call_args.args[0]
    assert req.full_url == "https://api/1"
    assert req.get_header("Authorization") == "Bearer tok"


def test_existing_file_is_not_downloaded(tmp_path):
    dest = tmp_path / "x.npz"
    dest.write_bytes(b"cached")
    with patch.object(assets._OPENER, "open") as opener:
        assets.ensure_asset("x.npz", str(dest))
    opener.assert_not_called()
