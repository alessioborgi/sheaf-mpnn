from __future__ import annotations

import pytest

from exp.benchmark.hparams import (
    OriginalHParams,
    OursHParams,
    validate_matched_architecture,
)


def _ours(stalk_dim=6, hidden_dim=128, num_layers=5) -> OursHParams:
    return OursHParams(
        model="nsd",
        best_params={
            "variant": "orthogonal",
            "stalk_dim": stalk_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "alpha": 0.5,
            "lr": 0.003,
        },
    )


def _original(d=6, layers=5, hidden_channels=128) -> OriginalHParams:
    return OriginalHParams(d=d, layers=layers, hidden_channels=hidden_channels)


def test_matched_architecture_passes_when_equal():
    validate_matched_architecture(
        _ours(), _original(), dataset="texas", variant="orthogonal"
    )


def test_matched_architecture_rejects_wrong_file_variant():
    # A copy-paste error in bestconf/ours/<dataset>/orthogonal.yaml whose
    # best_params say "diagonal" would otherwise silently benchmark the
    # wrong model under the "orthogonal" label.
    with pytest.raises(ValueError, match="best_params variant"):
        validate_matched_architecture(
            _ours(), _original(), dataset="texas", variant="diagonal"
        )


def test_matched_architecture_rejects_stalk_dim_mismatch():
    with pytest.raises(ValueError, match="stalk_dim"):
        validate_matched_architecture(
            _ours(stalk_dim=4),
            _original(d=6),
            dataset="texas",
            variant="orthogonal",
        )


def test_matched_architecture_rejects_hidden_dim_mismatch():
    with pytest.raises(ValueError, match="hidden_dim"):
        validate_matched_architecture(
            _ours(hidden_dim=64),
            _original(hidden_channels=128),
            dataset="texas",
            variant="orthogonal",
        )


def test_ours_hparams_to_config_routes_fields_by_dataclass():
    cfg = _ours().to_config(dataset="texas")
    assert cfg.dataset.name == "texas"
    assert cfg.model.variant == "orthogonal"
    assert cfg.model.stalk_dim == 6
    assert cfg.optim.lr == 0.003


def test_ours_hparams_rejects_best_params_missing_variant():
    with pytest.raises(ValueError, match="variant"):
        OursHParams(
            model="nsd",
            best_params={
                "stalk_dim": 6,
                "hidden_dim": 128,
                "num_layers": 5,
            },
        )
