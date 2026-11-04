# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Tests for the model registry."""

import pytest
import torch

from exp.config import ModelConfig, ModelType, RegConfig
from exp.registries.models import ModelEntry, ModelRegistry, model_registry
from sheaf_mpnn.nsd import NSDModel
from sheaf_mpnn.nsp import NSPModel
from sheaf_mpnn.sheaf_attn import SheafAttnModel

_REG = RegConfig()


def _make_cfg(**kwargs) -> ModelConfig:
    defaults = dict(
        type=ModelType.NSD,
        variant="general",
        stalk_dim=2,
        hidden_dim=4,
        num_layers=1,
        alpha=1.0,
        num_heads=1,
        leaky_relu_slope=0.2,
        clamp_val=10.0,
        orth_strategy="cayley",
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)  # ty: ignore[invalid-argument-type]


class TestModelRegistryRegistrations:
    def test_nsd_registered(self):
        assert "nsd" in model_registry

    def test_nsp_registered(self):
        assert "nsp" in model_registry

    def test_sheaf_attn_registered(self):
        assert "sheaf_attn" in model_registry

    def test_list_keys_contains_core_families(self):
        keys = set(model_registry.list_keys())
        assert {
            "nsd",
            "nsp",
            "sheaf_attn",
        }.issubset(keys)

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="unknown_model"):
            model_registry.get("unknown_model")


class TestModelRegistryBuild:
    def test_build_nsd_returns_nsd_model(self):
        cfg = _make_cfg(type=ModelType.NSD, variant="general")
        model = model_registry.build("nsd", 16, 7, cfg, _REG)
        assert isinstance(model, NSDModel)

    def test_build_nsp_returns_nsp_model(self):
        cfg = _make_cfg(type=ModelType.NSP, variant="general")
        model = model_registry.build("nsp", 16, 7, cfg, _REG)
        assert isinstance(model, NSPModel)

    def test_build_sheaf_attn_returns_sheaf_attn_model(self):
        cfg = _make_cfg(type=ModelType.SHEAF_ATTN, variant="sheafan")
        model = model_registry.build("sheaf_attn", 16, 7, cfg, _REG)
        assert isinstance(model, SheafAttnModel)

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

    def test_nsp_forward_shape(self):
        cfg = _make_cfg(
            type=ModelType.NSP, variant="general", stalk_dim=2, hidden_dim=4
        )  # noqa: E501
        model = model_registry.build("nsp", 8, 3, cfg, _REG)
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


class TestJointFamilyRegistryBuild:
    """The Hernandez Caralt et al. family must honour its ModelConfig knobs."""

    def test_build_id_sheaf_wires_alpha(self):
        from sheaf_mpnn.id_sheaf.id_sheaf_layers import IdentitySheafConv
        from sheaf_mpnn.id_sheaf.id_sheaf_model import IdentitySheafModel

        cfg = _make_cfg(type=ModelType.ID_SHEAF, alpha=0.5, learn_alpha=False)
        model = model_registry.build("id_sheaf", 16, 7, cfg, _REG)
        assert isinstance(model, IdentitySheafModel)
        for layer in model.layers:
            assert isinstance(layer, IdentitySheafConv)
            assert layer.alpha.item() == pytest.approx(0.5)
            assert not layer.alpha.requires_grad

    def test_build_joint_diff_wires_alpha_and_beta(self):
        from sheaf_mpnn.jointdiff.joint_layers import JointDiffusionSheafConv
        from sheaf_mpnn.jointdiff.joint_model import JointDiffusionModel

        cfg = _make_cfg(
            type=ModelType.JOINT_DIFF, alpha=0.5, beta=0.3, learn_alpha=False
        )
        model = model_registry.build("joint_diff", 16, 7, cfg, _REG)
        assert isinstance(model, JointDiffusionModel)
        for layer in model.layers:
            assert isinstance(layer, JointDiffusionSheafConv)
            assert layer.alpha.item() == pytest.approx(0.5)
            assert not layer.alpha.requires_grad
            assert layer.beta == pytest.approx(0.3)

    def test_build_rot_inv_wires_alpha_and_time_dep(self):
        from sheaf_mpnn.jointdiff.joint_model import RotationInvariantModel
        from sheaf_mpnn.jointdiff.rot_inv_layers import RotationInvariantSheafConv

        cfg = _make_cfg(type=ModelType.ROT_INV, alpha=2.0, time_dep=False)
        model = model_registry.build("rot_inv", 16, 7, cfg, _REG)
        assert isinstance(model, RotationInvariantModel)
        for layer in model.layers:
            assert isinstance(layer, RotationInvariantSheafConv)
            assert layer.alpha.item() == pytest.approx(2.0)
            assert layer.time_dep is False
