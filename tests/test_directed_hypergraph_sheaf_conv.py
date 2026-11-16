"""Independent dense and pairwise checks for magnetic incidence transport."""

import math

import pytest
import torch

from sheaf_mpnn import DHSNNVariant, DiagonalDHSNNConv, DiagonalHSNNConv
from sheaf_mpnn.directed_hypergraph_sheaf_conv import (
    ComplexFeatureNorm,
    complex_activation,
    complex_dropout,
)


def test_whitening_nearly_collinear_large_features_has_finite_gradients():
    torch.manual_seed(0)
    real = torch.randn(100, 8) * 1e4
    x = torch.complex(real, real + torch.randn_like(real) * 0.01).requires_grad_()
    norm = ComplexFeatureNorm(8)
    output = norm(x)
    assert torch.isfinite(output).all()
    output.abs().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in norm.parameters())


def dense_reference(index, roles, maps, n, m, q, tau=0):
    """Build dense incidence/degree with loops and eigh, independently of PyG."""
    d = maps.size(1)
    full = torch.diag_embed(maps) if maps.ndim == 2 else maps
    b = torch.zeros(m * d, n * d, dtype=torch.complex128, device=maps.device)
    blocks = [maps.new_zeros(d, d) for _ in range(n)]
    counts = [0] * m
    for k, (v, e) in enumerate(index.T.tolist()):
        phase = complex(math.cos(-2 * math.pi * q), math.sin(-2 * math.pi * q))
        b[e * d : (e + 1) * d, v * d : (v + 1) * d] = full[k] * (
            phase if roles[k] == -1 else 1
        )
        blocks[v] = blocks[v] + full[k].T @ full[k]
        counts[e] += 1
    degree = torch.block_diag(*blocks)
    inv_blocks = []
    for block in blocks:
        vals, vecs = torch.linalg.eigh(
            block + tau * torch.eye(d, device=maps.device, dtype=maps.dtype)
        )
        good = vals > vals.max() * d * torch.finfo(vals.dtype).eps
        inv = torch.where(good, torch.where(good, vals, 1).rsqrt(), 0)
        inv_blocks.append((vecs * inv) @ vecs.T)
    r = torch.block_diag(*inv_blocks).to(b.dtype)
    inv_e = maps.new_tensor([1 / max(c, 1) for c in counts]).repeat_interleave(d)
    a = b.mH @ (inv_e[:, None] * b)
    return r @ a @ r, degree.to(b.dtype) - a, r, degree.to(b.dtype), b


def fixture(variant, q=0.17, tau=0, device="cpu"):
    torch.manual_seed(29)
    layer = (
        variant.layer_class(2, 2, 3, context_dim=4, q=q, degree_regularization=tau)
        .double()
        .to(device)
    )
    index = torch.tensor([[0, 1, 2, 1, 3, 4], [0, 0, 0, 1, 1, 2]], device=device)
    roles = torch.tensor([-1, 1, 1, 1, -1, -1], device=device)
    maps = layer.get_restriction_maps(
        torch.randn(6, 4, dtype=torch.float64, device=device), index, num_hyperedges=4
    )
    x = torch.randn(6, 2, 2, dtype=torch.complex128, device=device)
    return layer, index, roles, maps, x


@pytest.mark.parametrize("variant", list(DHSNNVariant))
@pytest.mark.parametrize("q", [0.0, 0.17, 0.25])
@pytest.mark.parametrize("tau", [0.0, 0.2])
def test_dense_operator_and_spectrum(variant, q, tau):
    layer, index, roles, maps, x = fixture(variant, q, tau)
    p, lap, r, degree, b = dense_reference(index, roles, maps, 6, 4, q, tau)
    actual = layer.apply_diffusion(x, index, maps, roles, num_hyperedges=4)
    torch.testing.assert_close(actual.flatten(0, 1), p @ x.flatten(0, 1))
    torch.testing.assert_close(p, p.mH)
    torch.testing.assert_close(lap, lap.mH)
    assert torch.linalg.eigvalsh(lap).min() > -1e-10
    eig = torch.linalg.eigvalsh(p)
    assert eig.min() > -1e-10 and eig.max() < 1 + 1e-10
    torch.testing.assert_close(r @ lap @ r, r @ degree @ r - p)
    # Independent pairwise energy: sum ordered incidence differences / (2|e|).
    flat = x.flatten(0, 1)
    energy = x.real.new_zeros(())
    for e in range(4):
        members = index[0, index[1] == e].tolist()
        for v in members:
            for u in members:
                bv = b[e * 2 : (e + 1) * 2, v * 2 : (v + 1) * 2] @ x[v]
                bu = b[e * 2 : (e + 1) * 2, u * 2 : (u + 1) * 2] @ x[u]
                energy = energy + (bv - bu).abs().square().sum() / (2 * len(members))
    torch.testing.assert_close((flat.conj() * (lap @ flat)).sum().real, energy)


@pytest.mark.parametrize("variant", list(DHSNNVariant))
def test_reversal_periodicity_and_permutations(variant):
    layer, index, roles, maps, x = fixture(variant)
    p, *_ = dense_reference(index, roles, maps, 6, 4, layer.q)
    rev, *_ = dense_reference(index, -roles, maps, 6, 4, layer.q)
    torch.testing.assert_close(rev, p.conj())
    actual = layer.apply_diffusion(x, index, maps, roles, num_hyperedges=4)
    reversed_actual = layer.apply_diffusion(
        x.conj(), index, maps, -roles, num_hyperedges=4
    )
    torch.testing.assert_close(reversed_actual, actual.conj())
    layer.q += 1
    torch.testing.assert_close(
        layer.apply_diffusion(x, index, maps, roles, num_hyperedges=4), actual
    )
    order = torch.tensor([5, 2, 3, 0, 1, 4])
    torch.testing.assert_close(
        layer.apply_diffusion(
            x, index[:, order], maps[order], roles[order], num_hyperedges=4
        ),
        actual,
    )
    nodes = torch.tensor([4, 2, 0, 1, 5, 3])
    inverse = nodes.argsort()
    permuted = index.clone()
    permuted[0] = inverse[index[0]]
    torch.testing.assert_close(
        layer.apply_diffusion(x[nodes], permuted, maps, roles, num_hyperedges=4),
        actual[nodes],
    )


def test_phase_sign_and_undirected_limit():
    layer = DiagonalDHSNNConv(1, 2, 2, q=0.25).double()
    index = torch.tensor([[0, 1], [0, 0]])
    roles = torch.tensor([-1, 1])
    maps = torch.ones(2, 1, dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.complex128).view(2, 1, 2)
    actual = layer.apply_diffusion(eye, index, maps, roles).squeeze(1)
    torch.testing.assert_close(
        actual, torch.tensor([[0.5, 0.5j], [-0.5j, 0.5]], dtype=torch.complex128)
    )
    layer.q = 0
    real = torch.randn(2, 1, 2, dtype=torch.float64)
    hsnn = DiagonalHSNNConv(1, 2, 2).double()
    torch.testing.assert_close(
        layer.apply_diffusion(real.to(torch.complex128), index, maps, roles).real,
        hsnn.apply_diffusion(real, index, maps),
    )


@pytest.mark.parametrize("diagonal", [True, False])
def test_dense_gradients_and_gradcheck(diagonal):
    variant = DHSNNVariant.DIAGONAL if diagonal else DHSNNVariant.GENERAL
    layer, index, roles, maps, x = fixture(variant, tau=0.2)
    maps = maps.detach().requires_grad_()
    x.requires_grad_()
    actual = layer.apply_diffusion(x, index, maps, roles, num_hyperedges=4)
    p, *_ = dense_reference(index, roles, maps, 6, 4, layer.q, 0.2)
    expected = (p @ x.flatten(0, 1)).reshape_as(x)
    grads = torch.autograd.grad(
        actual.abs().square().sum(), (maps, x), retain_graph=True
    )
    oracle = torch.autograd.grad(expected.abs().square().sum(), (maps, x))
    for a, b in zip(grads, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-7)
    assert torch.autograd.gradcheck(
        lambda z, f: layer.apply_diffusion(z, index, f, roles, num_hyperedges=4),
        (x, maps),
        fast_mode=True,
    )


def test_empty_zero_single_sided_and_singular():
    layer = DiagonalDHSNNConv(2, 2, 3).double()
    x = torch.randn(3, 2, 2, dtype=torch.complex128)
    empty = torch.empty(2, 0, dtype=torch.long)
    roles = torch.empty(0, dtype=torch.long)
    out = layer.apply_diffusion(
        x, empty, torch.empty(0, 2, dtype=torch.float64), roles, num_hyperedges=2
    )
    assert out.count_nonzero() == 0
    index = torch.tensor([[0, 1], [0, 0]])
    maps = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    for role in (-1, 1):
        out = layer.apply_diffusion(x, index, maps, torch.full((2,), role))
        torch.testing.assert_close(out[:2, 0], x[:2, 0].mean(0).expand(2, 2))
        assert out[:, 1].count_nonzero() == 0
        assert out[2].count_nonzero() == 0
    singleton = torch.tensor([[0], [0]])
    torch.testing.assert_close(
        layer.apply_diffusion(x, singleton, maps[:1], torch.tensor([-1]))[0, 0], x[0, 0]
    )
    assert (
        layer.apply_diffusion(x, index, maps * 0, torch.tensor([-1, 1])).count_nonzero()
        == 0
    )


@pytest.mark.parametrize(
    "roles",
    [
        torch.tensor([0, 1]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([1]),
        torch.tensor([[1, -1]]),
    ],
)
def test_invalid_roles(roles):
    layer = DiagonalDHSNNConv(1, 1, 1)
    with pytest.raises(ValueError, match="incidence_roles"):
        layer.apply_diffusion(
            torch.ones(2, 1, 1, dtype=torch.complex64),
            torch.tensor([[0, 1], [0, 0]]),
            torch.ones(2, 1),
            roles,
        )


@pytest.mark.parametrize("q", [float("inf"), float("nan")])
def test_invalid_charge(q):
    with pytest.raises(ValueError, match="q"):
        DiagonalDHSNNConv(1, 1, 1, q=q)


def test_complex_activation_and_dropout():
    x = torch.tensor([1 + 2j, -1 + 3j, 0 + 5j])
    torch.testing.assert_close(
        complex_activation(x, "complex_relu"), torch.tensor([1 + 2j, 0j, 0 + 5j])
    )
    dropped = complex_dropout(
        torch.ones(100, dtype=torch.complex64) * (1 + 2j), 0.5, True
    )
    torch.testing.assert_close(dropped.imag, 2 * dropped.real)
    assert dropped.count_nonzero() not in (0, 100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("variant", list(DHSNNVariant))
@pytest.mark.parametrize("tau", [0.0, 0.1])
def test_cuda_forward_backward(variant, tau):
    cpu, index, roles, maps, x = fixture(variant, tau=tau)
    gpu, *_ = fixture(variant, tau=tau, device="cuda")
    gpu.load_state_dict(cpu.state_dict())
    maps = maps.detach().requires_grad_()
    x.requires_grad_()
    f_gpu = maps.detach().cuda().requires_grad_()
    x_gpu = x.detach().cuda().requires_grad_()
    a = cpu.apply_diffusion(x, index, maps, roles, num_hyperedges=4)
    b = gpu.apply_diffusion(x_gpu, index.cuda(), f_gpu, roles.cuda(), num_hyperedges=4)
    torch.testing.assert_close(a, b.cpu(), atol=1e-8, rtol=1e-6)
    ga = torch.autograd.grad(a.abs().square().sum(), (x, maps))
    gb = torch.autograd.grad(b.abs().square().sum(), (x_gpu, f_gpu))
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right.cpu(), atol=1e-7, rtol=1e-5)
