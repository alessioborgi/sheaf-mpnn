"""Directed model state retention, gradients, persistence and optimization."""

import io

import pytest
import torch
import torch.nn.functional as F

from sheaf_mpnn import DHSNNModel, DHSNNVariant


@pytest.mark.parametrize("variant", list(DHSNNVariant))
@pytest.mark.parametrize("q", [0.0, 0.25])
@pytest.mark.parametrize("activation", ["complex_relu", "componentwise_elu"])
def test_model_forward_backward_and_serialization(variant, q, activation):
    torch.manual_seed(10)
    model = DHSNNModel(
        3, 2, stalk_dim=2, hidden_dim=3, variant=variant, q=q, activation=activation
    ).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    states = []
    hooks = [
        layer.register_forward_hook(lambda module, inputs, out: states.append(out))
        for layer in model.layers
    ]
    logits = model(x, index, roles)
    for hook in hooks:
        hook.remove()
    assert logits.shape == (5, 2) and not logits.is_complex()
    assert all(state.is_complex() for state in states)
    assert any(state.imag.count_nonzero() > 0 for state in states)
    F.cross_entropy(logits, torch.tensor([0, 1, 0, 1, 0])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    model.eval()
    expected = model(x, index, roles)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model.reset_parameters()
    model.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(model(x, index, roles), expected)


def test_short_optimization_and_direction_response():
    torch.manual_seed(71)
    model = DHSNNModel(3, 2, stalk_dim=2, hidden_dim=8, num_layers=1, residual=True)
    index = torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    x = torch.randn(5, 3)
    y = torch.tensor([0, 0, 1, 1, 0])
    assert not torch.allclose(model(x, index, roles), model(x, index, -roles))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = F.cross_entropy(model(x, index, roles), y).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, index, roles), y)
        loss.backward()
        optimizer.step()
    assert F.cross_entropy(model(x, index, roles), y).item() < initial * 0.6


@pytest.mark.parametrize("variant", list(DHSNNVariant))
def test_light_preserves_forward_and_detaches_only_map_learners(variant):
    torch.manual_seed(19)
    full = DHSNNModel(3, 2, stalk_dim=2, hidden_dim=3, variant=variant)
    light = DHSNNModel(3, 2, stalk_dim=2, hidden_dim=3, variant=variant, light=True)
    light.load_state_dict(full.state_dict())
    x = torch.randn(5, 3, requires_grad=True)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, -1, 1, 1])
    output = light(x, index, roles)
    torch.testing.assert_close(output, full(x, index, roles))
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in light.named_parameters():
        if name.startswith("map_learners."):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
