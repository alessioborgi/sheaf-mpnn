# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch

from sheaf_mpnn.utils.orthogonal import attention_cayley, cayley, householder


def test_cayley_orthogonality():
    d = 4
    params = torch.randn(5, d * (d - 1) // 2)
    W = cayley(params, d)
    eye_mat = torch.eye(d).unsqueeze(0)
    assert torch.allclose(torch.matmul(W.transpose(-2, -1), W), eye_mat, atol=1e-5)


def test_cayley_identity():
    d = 3
    params = torch.zeros(1, d * (d - 1) // 2)
    W = cayley(params, d)
    assert torch.allclose(W, torch.eye(d).unsqueeze(0), atol=1e-6)


def test_cayley_clamping():
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
    d = 2
    raw = torch.ones(1, d * d)
    W = attention_cayley(raw, d)
    assert torch.allclose(W, torch.eye(d).unsqueeze(0), atol=1e-6)


def test_householder_orthogonality():
    d = 4
    params = torch.randn(5, d * d)
    W = householder(params, d)
    eye = torch.eye(d).unsqueeze(0)
    assert torch.allclose(torch.matmul(W.transpose(-2, -1), W), eye, atol=1e-5)


def test_householder_shape():
    d, batch_size = 3, 7
    params = torch.randn(batch_size, d * d)
    W = householder(params, d)
    assert W.shape == (batch_size, d, d)


def test_householder_stop_recursion_orthogonality():
    d = 4
    params = torch.randn(5, d * d)
    W = householder(params, d, stop_recursion=1)
    eye = torch.eye(d).unsqueeze(0)
    assert torch.allclose(torch.matmul(W.transpose(-2, -1), W), eye, atol=1e-5)


def test_householder_stop_recursion_shape():
    d, batch_size = 4, 6
    params = torch.randn(batch_size, d * d)
    W = householder(params, d, stop_recursion=0)
    assert W.shape == (batch_size, d, d)


def test_fasthpp_matches_householder_orgqr():
    """Fasthpp must compute the reference householder map exactly: same compact
    parameterization, product of the same reflections, no torch_householder.
    """
    import torch

    from sheaf_mpnn.utils.orthogonal import fasthpp, householder_orgqr

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for d in (2, 3, 4, 5, 8):
            torch.manual_seed(d)
            p = torch.randn(32, d * (d - 1) // 2)
            q_a, q_b = fasthpp(p, d), householder_orgqr(p, d)
            assert (q_a - q_b).abs().max() < 1e-12, d
            eye = torch.eye(d)
            assert (q_a.transpose(-1, -2) @ q_a - eye).abs().max() < 1e-12, d
    finally:
        torch.set_default_dtype(prev)


def test_build_orthogonal_all_strategies():
    """Every registered strategy yields orthogonal maps at its param count."""
    import torch

    from sheaf_mpnn.utils.orthogonal import (
        ORTH_STRATEGIES,
        build_orthogonal,
        orth_param_count,
    )

    torch.manual_seed(0)
    for d in (2, 3, 5):
        for strategy in ORTH_STRATEGIES:
            p = torch.randn(16, orth_param_count(strategy, d))
            q = build_orthogonal(p, d, strategy)
            eye = torch.eye(d)
            assert (q.transpose(-1, -2) @ q - eye).abs().max() < 1e-4, (strategy, d)
