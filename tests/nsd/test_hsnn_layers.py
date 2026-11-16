# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""HSNN restriction-map families and invariance properties."""

import pytest
import torch
from torch import nn

from sheaf_mpnn import (
    DiagonalHSNNConv,
    GeneralHSNNConv,
    LowRankHSNNConv,
    OrthogonalHSNNConv,
)

FAMILIES = [
    (DiagonalHSNNConv, {}),
    (GeneralHSNNConv, {}),
    (OrthogonalHSNNConv, {}),
    (OrthogonalHSNNConv, {"orth_strategy": "fasth"}),
    (OrthogonalHSNNConv, {"orth_strategy": "fasthpp"}),
    (OrthogonalHSNNConv, {"orth_strategy": "householder"}),
    (LowRankHSNNConv, {"rank": 2}),
]


@pytest.mark.parametrize("layer_class,kwargs", FAMILIES)
class TestHSNNFamilies:
    @pytest.fixture
    def setup(self, layer_class, kwargs):
        layer = layer_class(3, 2, 4, **kwargs).double()
        context = torch.randn(5, 6, dtype=torch.float64)
        stalk = torch.randn(5, 3, 2, dtype=torch.float64)
        index = torch.tensor([[0, 1, 2, 1, 3, 2, 0], [0, 0, 0, 1, 1, 2, 2]])
        return layer, context, stalk, index

    def test_forward_backward(self, setup):
        layer, context, stalk, index = setup
        context.requires_grad_()
        stalk.requires_grad_()
        out = layer(context, stalk, index, num_hyperedges=7)
        assert out.shape == stalk.shape and torch.isfinite(out).all()
        out.square().sum().backward()
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        assert context.grad is not None and context.grad.abs().sum() > 0
        assert stalk.grad is not None and torch.isfinite(stalk.grad).all()

    def test_node_edge_and_incidence_permutations(self, setup):
        layer, context, stalk, index = setup
        expected = layer(context, stalk, index, num_hyperedges=7)
        new_to_old = torch.randperm(5)
        old_to_new = new_to_old.argsort()
        permuted_index = torch.stack((old_to_new[index[0]], index[1]))
        actual = layer(
            context[new_to_old], stalk[new_to_old], permuted_index, num_hyperedges=7
        )
        torch.testing.assert_close(actual, expected[new_to_old], atol=1e-8, rtol=1e-7)
        edge_permutation = torch.randperm(7)
        columns = torch.randperm(index.size(1))
        permuted_index = torch.stack((index[0], edge_permutation[index[1]]))[:, columns]
        actual = layer(context, stalk, permuted_index, num_hyperedges=7)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)

    def test_forward_matches_public_transport_and_gradients(self, setup):
        """The optimized forward must retain the standalone API's semantics."""
        layer, context, stalk, index = setup
        context.requires_grad_()
        stalk.requires_grad_()
        layer.residual = True
        actual = layer(context, stalk, index, num_hyperedges=7)
        maps = layer.get_restriction_maps(context, index, num_hyperedges=7)
        z = layer.W1 @ stalk @ layer.W2
        expected = layer.sigma(
            layer.apply_diffusion(z, index, maps, num_hyperedges=7) + stalk
        )
        torch.testing.assert_close(actual, expected)
        inputs = (context, stalk, *layer.parameters())
        probe = torch.randn_like(stalk)
        actual_grad = torch.autograd.grad((actual * probe).sum(), inputs)
        expected_grad = torch.autograd.grad((expected * probe).sum(), inputs)
        for observed, reference in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(observed, reference)

    def test_no_forward_cache(self, setup):
        layer, context, stalk, index = setup
        first = layer(context, stalk, index)
        layer(context + 0.3, stalk + 0.1, index[:, :3], num_hyperedges=8)
        torch.testing.assert_close(layer(context, stalk, index), first)
        assert not torch.allclose(layer(context + 0.3, stalk, index), first)

    def test_empty_and_residual(self, setup):
        layer, context, stalk, _ = setup
        empty = torch.empty(2, 0, dtype=torch.long)
        out = layer(context, stalk, empty, num_hyperedges=8)
        torch.testing.assert_close(out, torch.zeros_like(stalk))
        layer.residual = True
        torch.testing.assert_close(layer(context, stalk, empty), layer.sigma(stalk))

    def test_stalk_dimension_one(self, layer_class, kwargs):
        kwargs = dict(kwargs)
        if "rank" in kwargs:
            kwargs["rank"] = 1
        layer = layer_class(1, 2, 2, **kwargs)
        context, stalk = torch.randn(3, 2), torch.randn(3, 1, 2)
        index = torch.tensor([[0, 1], [0, 0]])
        assert torch.isfinite(layer(context, stalk, index)).all()
        assert torch.isfinite(layer(context, stalk, index[:, :0])).all()


@pytest.mark.parametrize("strategy", ["cayley", "fasth", "fasthpp", "householder"])
def test_orthogonality(strategy):
    layer = OrthogonalHSNNConv(4, 2, 2, orth_strategy=strategy).double()
    index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    maps = layer.get_restriction_maps(torch.randn(3, 8).double(), index)
    identity = torch.eye(4, dtype=torch.float64).expand(3, -1, -1)
    torch.testing.assert_close(maps.transpose(-1, -2) @ maps, identity)


def test_low_rank_parameterization():
    layer = LowRankHSNNConv(4, 2, 2, rank=1).double()
    maps = layer.get_restriction_maps(
        torch.randn(3, 8).double(), torch.tensor([[0, 1, 2], [0, 0, 0]])
    )
    assert (torch.linalg.matrix_rank(maps) <= 1).all()


def test_update_formula_and_dropout():
    layer = DiagonalHSNNConv(2, 2, 2, dropout=1.0, residual=True)
    x = torch.randn(3, 2, 2)
    index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    torch.testing.assert_close(layer(x.flatten(1), x, index), nn.ELU()(x))
    layer.eval()
    maps = layer.get_restriction_maps(x.flatten(1), index)
    expected = layer.apply_diffusion(layer.W1 @ x @ layer.W2, index, maps)
    torch.testing.assert_close(layer(x.flatten(1), x, index), nn.ELU()(expected + x))


@pytest.mark.parametrize("regularization", [-1.0, float("nan"), float("inf")])
def test_invalid_regularization(regularization):
    with pytest.raises(ValueError, match="degree_regularization"):
        DiagonalHSNNConv(2, 2, 2, degree_regularization=regularization)


@pytest.mark.parametrize("rank", [0, -1, 4])
def test_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        LowRankHSNNConv(3, 2, 2, rank=rank)


def test_invalid_orthogonal_strategy():
    with pytest.raises(ValueError, match="orth_strategy"):
        OrthogonalHSNNConv(2, 2, 2, orth_strategy="unknown")  # ty: ignore[invalid-argument-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("layer_class,kwargs", FAMILIES)
@pytest.mark.parametrize("regularization", [0.0, 0.1])
def test_cuda_parity(layer_class, kwargs, regularization):
    cpu = layer_class(3, 2, 2, degree_regularization=regularization, **kwargs).double()
    gpu = (
        layer_class(3, 2, 2, degree_regularization=regularization, **kwargs)
        .double()
        .cuda()
    )
    gpu.load_state_dict(cpu.state_dict())
    context = torch.randn(3, 6, dtype=torch.float64)
    index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    x = torch.randn(3, 3, 2, dtype=torch.float64)
    first, second = cpu(context, x, index), gpu(context.cuda(), x.cuda(), index.cuda())
    torch.testing.assert_close(first, second.cpu(), atol=1e-8, rtol=1e-7)
    first.square().sum().backward()
    second.square().sum().backward()
    for a, b in zip(cpu.parameters(), gpu.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad.cpu(), atol=1e-7, rtol=1e-6)
