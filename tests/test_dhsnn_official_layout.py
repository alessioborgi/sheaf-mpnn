"""Check the official node-by-edge convention against incidence message passing."""

import pytest
import torch

from sheaf_mpnn import DHSNNVariant


@pytest.mark.parametrize("q", [0.0, 0.17, 0.25])
def test_official_incidence_layout_forward_and_gradient(q):
    """H has positive tail phase and transposed real blocks relative to B.

    This independently assembles the official symmetric block operator's
    algebraic form. It does not execute the upstream normalization routine
    (whose identity shift and detached gradients are separate differences).
    """
    torch.manual_seed(81)
    layer = DHSNNVariant.GENERAL.layer_class(
        2, 2, 3, context_dim=4, q=q, degree_regularization=0.2
    ).double()
    index = torch.tensor([[0, 1, 2, 1], [0, 0, 0, 1]])
    roles = torch.tensor([-1, 1, 1, -1])
    maps = torch.randn(4, 2, 2, dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 2, 2, dtype=torch.complex128, requires_grad=True)
    h = torch.zeros(6, 4, dtype=torch.complex128)
    degrees = [maps.new_zeros(2, 2) for _ in range(3)]
    counts = [0, 0]
    for k, (v, e) in enumerate(index.T.tolist()):
        phase = torch.exp(torch.tensor(2j * torch.pi * q)) if roles[k] == -1 else 1
        h[2 * v : 2 * v + 2, 2 * e : 2 * e + 2] = maps[k].T * phase
        degrees[v] = degrees[v] + maps[k].T @ maps[k]
        counts[e] += 1
    blocks = []
    for degree in degrees:
        vals, vecs = torch.linalg.eigh(degree + 0.2 * torch.eye(2))
        blocks.append((vecs * vals.rsqrt()) @ vecs.T)
    r = torch.block_diag(*blocks).to(torch.complex128)
    inv_e = maps.new_tensor([1 / c for c in counts]).repeat_interleave(2)
    expected = (r @ h @ torch.diag(inv_e).to(h.dtype) @ h.mH @ r) @ x.flatten(0, 1)
    actual = layer.apply_diffusion(x, index, maps, roles, num_hyperedges=2)
    torch.testing.assert_close(actual.flatten(0, 1), expected, atol=2e-7, rtol=2e-7)
    actual_grad = torch.autograd.grad(
        actual.abs().square().sum(), (x, maps), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (x, maps))
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
