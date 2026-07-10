# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch

from sheaf_mpnn.utils import normalization as _norm
from sheaf_mpnn.utils.normalization import (
    apply_general_norm,
    apply_low_rank_norm,
    apply_orthogonal_norm,
)


def test_apply_orthogonal_norm_matches_degree_formula():
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [1, 2, 2, 2],
        ],
        dtype=torch.long,
    )
    cross_map = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ]
    )

    norm_self, norm_cross = apply_orthogonal_norm(
        cross_map,
        edge_index,
        num_nodes=3,
    )

    src, dst = edge_index
    deg = torch.tensor([0.0, 1.0, 3.0])
    # Augmented normalization: (deg + 1)^{-1/2}, finite for isolated nodes.
    deg_sqrt_inv = (deg + 1).pow(-0.5)
    expected_self = (deg_sqrt_inv[dst] ** 2).view(-1, 1, 1)
    expected_cross = (deg_sqrt_inv[dst] * deg_sqrt_inv[src]).view(-1, 1, 1) * cross_map

    assert torch.allclose(norm_self, expected_self)
    assert torch.allclose(norm_cross, expected_cross)


def test_apply_general_norm_rejects_nonfinite_self_map(monkeypatch):
    d, num_nodes = 2, 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    self_map = torch.eye(d).repeat(1, 1, 1)
    cross_map = torch.eye(d).repeat(1, 1, 1)
    self_map[0, 0, 0] = torch.inf

    def fake_matrix_pow(matrices, _p):
        return torch.eye(d, dtype=matrices.dtype, device=matrices.device).repeat(
            matrices.size(0), 1, 1
        )

    monkeypatch.setattr(_norm, "batched_sym_matrix_pow_svd", fake_matrix_pow)

    with pytest.raises(RuntimeError, match="norm_self contains non-finite"):
        apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            stalk_dim=d,
            training=False,
        )


def test_apply_general_norm_rejects_nonfinite_cross_map(monkeypatch):
    d, num_nodes = 2, 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    self_map = torch.eye(d).repeat(1, 1, 1)
    cross_map = torch.eye(d).repeat(1, 1, 1)
    cross_map[0, 0, 0] = torch.inf

    def fake_matrix_pow(matrices, _p):
        return torch.eye(d, dtype=matrices.dtype, device=matrices.device).repeat(
            matrices.size(0), 1, 1
        )

    monkeypatch.setattr(_norm, "batched_sym_matrix_pow_svd", fake_matrix_pow)

    with pytest.raises(RuntimeError, match="norm_cross contains non-finite"):
        apply_general_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            stalk_dim=d,
            training=False,
        )


def _fixed_channel_collision_maps():
    """Block maps whose fixed-channel degree (deg+1=2) exactly ties a learned
    singular value -- the case whose SVD backward is non-finite if the fixed
    channels enter the decomposition.
    """
    d, k = 2, 1
    edge_index = torch.tensor([[0, 1], [1, 0]])
    self_map = torch.zeros(2, d + k, d + k)
    self_map[:, :d, :d] = torch.diag(torch.tensor([1.0, 0.25]))
    self_map[:, d, d] = 1.0
    return self_map, self_map.clone(), edge_index


def test_apply_general_norm_fixed_channel_gradients_finite():
    self_map, cross_map, edge_index = _fixed_channel_collision_maps()
    self_map.requires_grad_(True)
    cross_map.requires_grad_(True)

    norm_self, norm_cross = apply_general_norm(
        self_map,
        cross_map,
        edge_index,
        num_nodes=2,
        stalk_dim=3,
        training=False,
        learned_dim=2,
    )
    (norm_self.sum() + norm_cross.sum()).backward()

    assert self_map.grad is not None and torch.isfinite(self_map.grad).all()
    assert cross_map.grad is not None and torch.isfinite(cross_map.grad).all()


def test_apply_general_norm_learned_dim_matches_joint_forward():
    # Block-diagonal (D+I)^{-1/2} equals the joint computation in eval mode,
    # so excluding fixed channels from the SVD must not change the operator.
    self_map, cross_map, edge_index = _fixed_channel_collision_maps()

    joint = apply_general_norm(
        self_map, cross_map, edge_index, 2, stalk_dim=3, training=False
    )
    block = apply_general_norm(
        self_map, cross_map, edge_index, 2, stalk_dim=3, training=False, learned_dim=2
    )
    assert torch.allclose(joint[0], block[0], atol=1e-6)
    assert torch.allclose(joint[1], block[1], atol=1e-6)


def test_batched_sym_matrix_pow_svd_gradcheck():
    torch.manual_seed(0)
    a = torch.randn(3, 4, 4, dtype=torch.float64)
    spd = a @ a.transpose(-1, -2) + 2 * torch.eye(4, dtype=torch.float64)
    spd.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda m: _norm.batched_sym_matrix_pow_svd(m, -0.5),
        (spd,),
        eps=1e-6,
        atol=1e-8,
    )


def test_batched_sym_matrix_pow_svd_degenerate_backward():
    # Repeated singular values: generic SVD autograd returns NaN; the
    # Daleckii-Krein backward gives the finite limit g'(1) = -0.5.
    eye = torch.eye(4, dtype=torch.float64).expand(2, 4, 4).clone()
    eye.requires_grad_(True)
    out = _norm.batched_sym_matrix_pow_svd(eye, -0.5)
    out.backward(torch.ones_like(out))

    assert eye.grad is not None
    assert torch.isfinite(eye.grad).all()
    assert torch.allclose(eye.grad, torch.full_like(eye.grad, -0.5))


def test_apply_low_rank_norm_rejects_nonfinite_self_map(monkeypatch):
    d, num_nodes = 2, 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    self_map = torch.eye(d).repeat(1, 1, 1)
    cross_map = torch.eye(d).repeat(1, 1, 1)
    self_map[0, 0, 0] = torch.inf

    def fake_matrix_pow(matrices, _p):
        return torch.eye(d, dtype=matrices.dtype, device=matrices.device).repeat(
            matrices.size(0), 1, 1
        )

    monkeypatch.setattr(_norm, "batched_sym_matrix_pow_svd", fake_matrix_pow)

    with pytest.raises(RuntimeError, match="norm_self contains non-finite"):
        apply_low_rank_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            stalk_dim=d,
            training=False,
        )


def test_apply_low_rank_norm_rejects_nonfinite_cross_map(monkeypatch):
    d, num_nodes = 2, 2
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    self_map = torch.eye(d).repeat(1, 1, 1)
    cross_map = torch.eye(d).repeat(1, 1, 1)
    cross_map[0, 0, 0] = torch.inf

    def fake_matrix_pow(matrices, _p):
        return torch.eye(d, dtype=matrices.dtype, device=matrices.device).repeat(
            matrices.size(0), 1, 1
        )

    monkeypatch.setattr(_norm, "batched_sym_matrix_pow_svd", fake_matrix_pow)

    with pytest.raises(RuntimeError, match="norm_cross contains non-finite"):
        apply_low_rank_norm(
            self_map,
            cross_map,
            edge_index,
            num_nodes,
            stalk_dim=d,
            training=False,
        )
