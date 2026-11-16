"""Directed layer families, input validation and hyperedge permutations."""

import pytest
import torch

from sheaf_mpnn import DHSNNVariant, DiagonalDHSNNConv


@pytest.mark.parametrize("variant", list(DHSNNVariant))
def test_maps_and_hyperedge_permutation(variant):
    torch.manual_seed(4)
    layer = variant.layer_class(3, 2, 2, context_dim=12).double()
    x = torch.randn(4, 3, 2, dtype=torch.complex128)
    context = torch.cat((x.flatten(1).real, x.flatten(1).imag), -1)
    index = torch.tensor([[0, 1, 2, 1, 3], [0, 0, 0, 2, 2]])
    roles = torch.tensor([-1, 1, 1, -1, 1])
    maps = layer.get_restriction_maps(context, index, num_hyperedges=4)
    if variant == DHSNNVariant.ORTHOGONAL:
        torch.testing.assert_close(
            maps.mT @ maps, torch.eye(3, dtype=maps.dtype).expand(5, 3, 3)
        )
    if variant == DHSNNVariant.LOW_RANK:
        assert (torch.linalg.matrix_rank(maps) <= 1).all()
    out = layer(context, x, index, roles, num_hyperedges=4)
    permuted = index.clone()
    permuted[1] = torch.tensor([2, 3, 0, 1])[index[1]]
    torch.testing.assert_close(
        out, layer(context, x, permuted, roles, num_hyperedges=4)
    )
    out.abs().square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()
    )


@pytest.mark.parametrize("bad", ["duplicate", "negative", "range", "float", "shape"])
def test_bad_topology(bad):
    layer = DiagonalDHSNNConv(1, 1, 1)
    index = torch.tensor([[0, 1], [0, 0]])
    if bad == "duplicate":
        index[0, 1] = 0
    elif bad == "negative":
        index[0, 0] = -1
    elif bad == "range":
        index[0, 1] = 2
    elif bad == "float":
        index = index.float()
    else:
        index = index.flatten()
    with pytest.raises(ValueError):
        layer.apply_diffusion(
            torch.ones(2, 1, 1, dtype=torch.complex64),
            index,
            torch.ones(2, 1),
            torch.tensor([-1, 1]),
        )


@pytest.mark.parametrize("bad", ["real_stalk", "map_shape", "map_dtype", "map_nan"])
def test_bad_states_and_maps(bad):
    layer = DiagonalDHSNNConv(1, 1, 1)
    x = torch.ones(2, 1, 1, dtype=torch.complex64)
    maps = torch.ones(2, 1)
    if bad == "real_stalk":
        x = x.real
    elif bad == "map_shape":
        maps = maps[:, 0]
    elif bad == "map_dtype":
        maps = maps.double()
    else:
        maps[0] = float("nan")
    with pytest.raises(ValueError):
        layer.apply_diffusion(
            x, torch.tensor([[0, 1], [0, 0]]), maps, torch.tensor([-1, 1])
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("variant", list(DHSNNVariant))
@pytest.mark.parametrize("tau", [0.0, 0.1])
def test_learned_layer_cuda_parameter_gradients(variant, tau):
    torch.manual_seed(17)
    cpu = variant.layer_class(
        2, 2, 3, context_dim=8, degree_regularization=tau
    ).double()
    gpu = (
        variant.layer_class(2, 2, 3, context_dim=8, degree_regularization=tau)
        .double()
        .cuda()
    )
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(5, 2, 2, dtype=torch.complex128)
    context = torch.cat((x.flatten(1).real, x.flatten(1).imag), -1)
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 1]])
    roles = torch.tensor([-1, 1, 1, 1, -1, 1])
    a = cpu(context, x, index, roles)
    b = gpu(context.cuda(), x.cuda(), index.cuda(), roles.cuda())
    torch.testing.assert_close(a, b.cpu(), atol=1e-8, rtol=1e-6)
    a.abs().square().sum().backward()
    b.abs().square().sum().backward()
    for left, right in zip(cpu.parameters(), gpu.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad.cpu(), atol=1e-7, rtol=1e-5)
