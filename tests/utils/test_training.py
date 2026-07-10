# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch

from sheaf_mpnn.utils.training import setup_torch


def test_setup_torch_execution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

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
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    setup_torch(precision="high", seed=456)
    assert torch.initial_seed() == 456
