# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the model registry."""

import pytest
import torch

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries.models import ModelEntry, ModelRegistry, model_registry
from sheaf_mpnn.nsd import NSDModel

_REG = RegConfig()


def _make_cfg(**kwargs) -> ModelConfig:
    defaults = dict(
        type=ModelType.NSD,
        variant="general",
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        alpha=1.0,
        orth_strategy="cayley",
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]


class TestModelRegistryRegistrations:
    def test_nsd_registered(self):
        assert "nsd" in model_registry

    def test_list_keys_contains_nsd(self):
        keys = set(model_registry.list_keys())
        assert "nsd" in keys

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="unknown_model"):
            model_registry.get("unknown_model")


class TestModelRegistryBuild:
    def test_build_nsd_returns_nsd_model(self):
        cfg = _make_cfg(type=ModelType.NSD, variant="general")
        model = model_registry.build("nsd", 16, 7, cfg, _REG)
        assert isinstance(model, NSDModel)

    def test_build_unknown_raises(self):
        cfg = _make_cfg()
        with pytest.raises(KeyError):
            model_registry.build("unknown", 16, 7, cfg, _REG)

    def test_build_respects_stalk_dim(self):
        cfg = _make_cfg(stalk_dim=3, hidden_dim=8)
        model = model_registry.build("nsd", 16, 7, cfg, _REG)
        assert model.stalk_dim == 3

    def test_build_respects_num_layers(self):
        cfg = _make_cfg(num_layers=3)
        model = model_registry.build("nsd", 16, 7, cfg, _REG)
        assert isinstance(model, NSDModel) and len(model.layers) == 3

    def test_nsd_forward_shape(self):
        cfg = _make_cfg(stalk_dim=2, hidden_dim=4, num_layers=1)
        model = model_registry.build("nsd", 8, 3, cfg, _REG)
        n = 10
        x = torch.randn(n, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (n, 3)


class TestModelRegistryCustomEntry:
    """Verify registry entries can be swapped in tests (the key testability win)."""

    def test_custom_entry_is_used(self):
        class _Dummy(torch.nn.Module):
            def forward(self, x, edge_index):
                return x

        registry = ModelRegistry()
        registry.register("dummy", ModelEntry(factory=lambda i, o, c, r: _Dummy()))
        model = registry.build("dummy", 8, 3, _make_cfg(), _REG)
        assert isinstance(model, _Dummy)
