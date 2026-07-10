# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò


import pytest
import torch

from sheaf_mpnn.utils import attention_cayley, cayley, setup_torch


def test_cayley_orthogonality():
    d = 4
    params = torch.randn(5, d * (d - 1) // 2)
    W = cayley(params, d)
    # Check W^T W = Identity
    eye_mat = torch.eye(d).unsqueeze(0)
    assert torch.allclose(torch.matmul(W.transpose(-2, -1), W), eye_mat, atol=1e-5)


def test_cayley_identity():
    # Zero params should produce identity
    d = 3
    params = torch.zeros(1, d * (d - 1) // 2)
    W = cayley(params, d)
    assert torch.allclose(W, torch.eye(d).unsqueeze(0), atol=1e-6)


def test_cayley_clamping():
    # Large params should be clamped
    d = 2
    params = torch.tensor([[100.0]])
    W1 = cayley(params, d, clamp_val=10.0)
    W2 = cayley(torch.tensor([[10.0]]), d)
    assert torch.allclose(W1, W2)


def test_attention_cayley_orthogonality():
    d = 3
    raw = torch.randn(5, d * d)
    W = attention_cayley(raw, d)
    eye_mat = torch.eye(d).unsqueeze(0)
    assert torch.allclose(torch.matmul(W.transpose(-2, -1), W), eye_mat, atol=1e-5)


def test_attention_cayley_constant():
    # Constant input should result in symmetric softmax, thus zero
    # skew-symmetric part, thus identity.
    d = 2
    raw = torch.ones(1, d * d)
    W = attention_cayley(raw, d)
    assert torch.allclose(W, torch.eye(d).unsqueeze(0), atol=1e-6)


def test_setup_torch_execution(monkeypatch):
    # Mocking to avoid actual hardware-dependent calls and cover all branches
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Force mps to be present for coverage
    class MockMPS:
        def is_available(self):
            return True

    monkeypatch.setattr(torch.backends, "mps", MockMPS(), raising=False)

    setup_torch(precision="medium", seed=123)
    assert torch.initial_seed() == 123


@pytest.mark.parametrize("prec", ["highest", "high", "medium"])
def test_setup_torch_precisions(prec, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    setup_torch(precision=prec, seed=42)
    assert torch.get_float32_matmul_precision() == prec


def test_setup_torch_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    # Mock backend flags to avoid real side effects in test
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    setup_torch(precision="high", seed=456)
    assert torch.initial_seed() == 456
