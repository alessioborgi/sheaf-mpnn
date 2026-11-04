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


def test_detach_degree_diagonal_norm():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    self_map = torch.tensor([[1.5, 2.5], [2.0, 3.0]], requires_grad=True)
    cross_map = torch.tensor([[0.5, 0.5], [0.6, 0.6]], requires_grad=True)

    # 1. Run with detach_degree=True
    norm_self, norm_cross = _norm.apply_diagonal_norm(
        self_map, cross_map, edge_index, num_nodes=2, detach_degree=True
    )
    loss = norm_self.sum() + norm_cross.sum()
    loss.backward()
    assert self_map.grad is not None
    grad_self_detached = self_map.grad.clone()

    # 2. Reset gradients
    self_map.grad.zero_()

    # 3. Run with detach_degree=False
    norm_self_f, norm_cross_f = _norm.apply_diagonal_norm(
        self_map, cross_map, edge_index, num_nodes=2, detach_degree=False
    )
    loss_f = norm_self_f.sum() + norm_cross_f.sum()
    loss_f.backward()
    assert self_map.grad is not None
    grad_self_full = self_map.grad.clone()

    # Gradients w.r.t self_map must differ: degree is detached in the first run
    assert not torch.allclose(grad_self_detached, grad_self_full)


def test_detach_degree_orthogonal_norm():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    cross_map = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]]], requires_grad=True
    )
    self_diag = torch.tensor([[1.5, 2.5], [2.0, 3.0]], requires_grad=True)

    # 1. Run with detach_degree=True
    norm_self, norm_cross = _norm.apply_orthogonal_norm(
        cross_map, edge_index, num_nodes=2, self_diag=self_diag, detach_degree=True
    )
    loss = norm_self.sum() + norm_cross.sum()
    loss.backward()
    assert self_diag.grad is not None
    grad_self_diag_detached = self_diag.grad.clone()

    # 2. Reset gradients
    self_diag.grad.zero_()

    # 3. Run with detach_degree=False
    norm_self_f, norm_cross_f = _norm.apply_orthogonal_norm(
        cross_map, edge_index, num_nodes=2, self_diag=self_diag, detach_degree=False
    )
    loss_f = norm_self_f.sum() + norm_cross_f.sum()
    loss_f.backward()
    assert self_diag.grad is not None
    grad_self_diag_full = self_diag.grad.clone()

    assert not torch.allclose(grad_self_diag_detached, grad_self_diag_full)


def test_detach_degree_general_norm():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    self_map = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)
    cross_map = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)

    # 1. Run with detach_degree=True
    norm_self, norm_cross = _norm.apply_general_norm(
        self_map,
        cross_map,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=True,
    )
    loss = norm_self.sum() + norm_cross.sum()
    loss.backward()
    assert self_map.grad is not None
    grad_self_detached = self_map.grad.clone()

    # 2. Reset gradients
    self_map.grad.zero_()

    # 3. Run with detach_degree=False
    norm_self_f, norm_cross_f = _norm.apply_general_norm(
        self_map,
        cross_map,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=False,
    )
    loss_f = norm_self_f.sum() + norm_cross_f.sum()
    loss_f.backward()
    assert self_map.grad is not None
    grad_self_full = self_map.grad.clone()

    assert not torch.allclose(grad_self_detached, grad_self_full)


def test_detach_degree_low_rank_norm():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    self_map = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)
    cross_map = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)

    # 1. Run with detach_degree=True
    norm_self, norm_cross = _norm.apply_low_rank_norm(
        self_map,
        cross_map,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=True,
    )
    loss = norm_self.sum() + norm_cross.sum()
    loss.backward()
    assert self_map.grad is not None
    grad_self_detached = self_map.grad.clone()

    # 2. Reset gradients
    self_map.grad.zero_()

    # 3. Run with detach_degree=False
    norm_self_f, norm_cross_f = _norm.apply_low_rank_norm(
        self_map,
        cross_map,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=False,
    )
    loss_f = norm_self_f.sum() + norm_cross_f.sum()
    loss_f.backward()
    assert self_map.grad is not None
    grad_self_full = self_map.grad.clone()

    assert not torch.allclose(grad_self_detached, grad_self_full)


def test_detach_degree_directed_norms():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    dst_map = torch.tensor([[1.5, 2.5], [2.0, 3.0]], requires_grad=True)
    src_map = torch.tensor([[1.0, 2.0], [2.5, 3.5]], requires_grad=True)
    cross_map = torch.tensor([[0.5, 0.5], [0.6, 0.6]], requires_grad=True)

    # 1. Test apply_directed_diagonal_norm
    norm_fwd, norm_bwd = _norm.apply_directed_diagonal_norm(
        dst_map, src_map, cross_map, edge_index, num_nodes=2, detach_degree=True
    )
    (norm_fwd.sum() + norm_bwd.sum()).backward()
    assert dst_map.grad is None
    assert src_map.grad is None

    norm_fwd_f, norm_bwd_f = _norm.apply_directed_diagonal_norm(
        dst_map, src_map, cross_map, edge_index, num_nodes=2, detach_degree=False
    )
    (norm_fwd_f.sum() + norm_bwd_f.sum()).backward()
    assert dst_map.grad is not None
    assert src_map.grad is not None

    # 2. Test apply_directed_general_norm
    dst_map_mat = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)
    src_map_mat = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)
    cross_map_mat = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)

    norm_fwd_g, norm_bwd_g = _norm.apply_directed_general_norm(
        dst_map_mat,
        src_map_mat,
        cross_map_mat,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=True,
    )
    (norm_fwd_g.sum() + norm_bwd_g.sum()).backward()
    assert dst_map_mat.grad is None
    assert src_map_mat.grad is None

    norm_fwd_gf, norm_bwd_gf = _norm.apply_directed_general_norm(
        dst_map_mat,
        src_map_mat,
        cross_map_mat,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=False,
    )
    (norm_fwd_gf.sum() + norm_bwd_gf.sum()).backward()
    assert dst_map_mat.grad is not None
    assert src_map_mat.grad is not None

    # 3. Test apply_directed_low_rank_norm
    dst_map_lr = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)
    src_map_lr = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)
    cross_map_lr = torch.eye(2).repeat(2, 1, 1).requires_grad_(True)

    norm_fwd_lr, norm_bwd_lr = _norm.apply_directed_low_rank_norm(
        dst_map_lr,
        src_map_lr,
        cross_map_lr,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=True,
    )
    (norm_fwd_lr.sum() + norm_bwd_lr.sum()).backward()
    assert dst_map_lr.grad is None
    assert src_map_lr.grad is None

    norm_fwd_lrf, norm_bwd_lrf = _norm.apply_directed_low_rank_norm(
        dst_map_lr,
        src_map_lr,
        cross_map_lr,
        edge_index,
        num_nodes=2,
        stalk_dim=2,
        training=False,
        detach_degree=False,
    )
    (norm_fwd_lrf.sum() + norm_bwd_lrf.sum()).backward()
    assert dst_map_lr.grad is not None
    assert src_map_lr.grad is not None
