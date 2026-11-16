# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Public HSNN model API, training, serialization and input validation."""

import io

import pytest
import torch
import torch.nn.functional as F

import sheaf_mpnn
from sheaf_mpnn import HSNNModel, HSNNVariant
from sheaf_mpnn.nsd import HSNNModel as ExportedModel


@pytest.mark.parametrize("variant", list(HSNNVariant))
def test_model_train_and_roundtrip(variant):
    model = HSNNModel(3, 2, stalk_dim=3, hidden_dim=3, variant=variant)
    x = torch.randn(6, 3)
    index = torch.tensor([[0, 1, 2, 1, 3, 4, 0, 4], [0, 0, 0, 1, 1, 1, 2, 2]])
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    logits = model(x, index, num_hyperedges=7)
    assert logits.shape == (6, 2)
    F.cross_entropy(logits, labels).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    model.eval()
    before = model(x, index)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = HSNNModel(3, 2, stalk_dim=3, hidden_dim=3, variant=variant).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x, index), before)
    model.reset_parameters()
    assert not torch.allclose(model(x, index), before)


def test_public_exports():
    assert HSNNModel is ExportedModel
    for name in (
        "BaseHypergraphSheafConv",
        "DiagonalHSNNConv",
        "GeneralHSNNConv",
        "OrthogonalHSNNConv",
        "LowRankHSNNConv",
        "HSNNModel",
        "HSNNVariant",
    ):
        assert name in sheaf_mpnn.__all__
        assert hasattr(sheaf_mpnn, name)


def test_eval_dropout_and_double():
    model = HSNNModel(3, 2, dropout=0.5, input_dropout=0.5).double().eval()
    x = torch.randn(5, 3).double()
    index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    torch.testing.assert_close(model(x, index), model(x, index))
    assert model(x, index).dtype == torch.float64


@pytest.mark.parametrize("variant", list(HSNNVariant))
def test_empty_model(variant):
    model = HSNNModel(3, 2, variant=variant)
    out = model(
        torch.randn(5, 3), torch.empty(2, 0, dtype=torch.long), num_hyperedges=8
    )
    assert out.shape == (5, 2) and torch.isfinite(out).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"stalk_dim": 0}, {"num_layers": 0}, {"hidden_dim": -1}, {"variant": "diagonal"}],
)
def test_invalid_model_parameters(kwargs):
    with pytest.raises(ValueError):
        HSNNModel(3, 2, **kwargs)


@pytest.mark.parametrize(
    "x", [torch.randn(4, 2), torch.empty(0, 3), torch.ones(4, 3).long()]
)
def test_invalid_features(x):
    model = HSNNModel(3, 2)
    with pytest.raises(ValueError):
        model(x, torch.empty(2, 0, dtype=torch.long))


@pytest.mark.parametrize("variant", list(HSNNVariant))
def test_short_training(variant):
    model = HSNNModel(2, 2, stalk_dim=2, hidden_dim=3, num_layers=1, variant=variant)
    x = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = F.cross_entropy(model(x, index), y).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, index), y)
        loss.backward()
        optimizer.step()
    final = F.cross_entropy(model(x, index), y)
    assert torch.isfinite(final) and final.item() < initial
