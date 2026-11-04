# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Regression guard: parity with the reference NSD of Bodnar et al.

Each test pins one structural property whose loss caused a systematic
accuracy gap: augmented (D+1)/(D+I) normalization, map-scale sensitivity,
train/eval operator consistency, and symmetrized loop-free datasets.
The reference operator is built densely here; no reference repo needed.
"""

from typing import Any

import pytest
import torch
from torch import nn

from sheaf_mpnn.nsd.nsd_layers import DiagonalNSDConv, GeneralNSDConv
from sheaf_mpnn.utils.normalization import apply_general_norm

# Undirected 5-node graph, both directions, no self-loops.
UND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 4)]
N, D, F_DIM = 5, 2, 3


@pytest.fixture(autouse=True)
def _float64_default():
    """Exact-arithmetic comparisons need float64; restore float32 afterwards."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def both_dir_edge_index():
    e = torch.tensor(UND_EDGES).t()
    return torch.cat([e, e.flip(0)], dim=1)


def dense_reference_laplacian(edge_index, maps, augmented=True):
    """Bodnar's normalized sheaf Laplacian, built densely.

    maps[e] = restriction map of the SOURCE node of directed edge e
    (generated from cat([x_src, x_dst]) in the reference).
    L = Dhat^{-1/2} (D - A) Dhat^{-1/2}, Dhat = D + I if augmented.
    """
    E = edge_index.size(1)
    L = torch.zeros(N * D, N * D)
    Ddiag = torch.zeros(N * D, N * D)
    for e in range(E):
        u = int(edge_index[0, e])
        Ddiag[u * D : (u + 1) * D, u * D : (u + 1) * D] += maps[e].T @ maps[e]
    for e in range(E):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        rev = [(int(edge_index[0, k]), int(edge_index[1, k])) for k in range(E)].index(
            (v, u)
        )
        L[u * D : (u + 1) * D, v * D : (v + 1) * D] = -maps[e].T @ maps[rev]
    L = L + Ddiag
    Dhat = Ddiag + (torch.eye(N * D) if augmented else 0)
    dinv = torch.zeros_like(Dhat)
    for u in range(N):
        blk = Dhat[u * D : (u + 1) * D, u * D : (u + 1) * D]
        vals, vecs = torch.linalg.eigh(blk)
        dinv[u * D : (u + 1) * D, u * D : (u + 1) * D] = (
            vecs @ torch.diag(vals.clamp(min=1e-12).pow(-0.5)) @ vecs.T
        )
    return dinv @ L @ dinv


def drive_layer_with_maps(layer, edge_index, maps, x_stalk):
    """Run the user's layer with injected maps; return its Lx."""
    layer.W1.data = torch.eye(D)
    layer.W2.data = torch.eye(F_DIM)
    layer.alpha.data = torch.tensor(1.0)
    layer.sigma = nn.Identity()

    pairs = [
        (int(edge_index[0, k]), int(edge_index[1, k]))
        for k in range(edge_index.size(1))
    ]

    def fake_products(x_feat, ei):
        E2 = ei.size(1)
        self_m = torch.zeros(E2, D, D)
        cross_m = torch.zeros(E2, D, D)
        for k in range(E2):
            s, t = int(ei[0, k]), int(ei[1, k])
            if s == t:
                # Appended self-loop: mimic the REAL map generator, which at
                # the gain=0.01 init emits near-zero maps (not identity).
                # After the fix (augmentation inside the normalization,
                # add_self_loops=False) this branch is never taken.
                F_t = F_s = torch.full((D, D), 1e-4)
            else:
                F_t = maps[
                    pairs.index((t, s))
                ]  # dst map: generated from cat([x_dst, x_src])
                F_s = maps[pairs.index((s, t))]
            self_m[k] = F_t.T @ F_t
            cross_m[k] = F_t.T @ F_s
        return self_m, cross_m

    layer.get_map_products = fake_products
    layer.eval()
    with torch.no_grad():
        out = layer(torch.zeros(N, D * F_DIM), x_stalk, edge_index)
    return x_stalk - out  # update is x - alpha*sigma(Lx) with alpha=1, sigma=id


def test_general_matches_reference_augmented_laplacian():
    """User GeneralNSDConv must reproduce Bodnar's augmented=True operator."""
    torch.manual_seed(0)
    ei = both_dir_edge_index()
    maps = torch.randn(ei.size(1), D, D) * 0.5
    x = torch.randn(N, D, F_DIM)

    y_ref = (
        dense_reference_laplacian(ei, maps, augmented=True) @ x.reshape(N * D, F_DIM)
    ).reshape(N, D, F_DIM)

    layer = GeneralNSDConv(D, F_DIM, 8, context_dim=D * F_DIM)
    y_user = drive_layer_with_maps(layer, ei, maps, x)

    rel = (y_user - y_ref).norm() / y_ref.norm()
    assert rel < 1e-8, (
        f"rel Frobenius error vs reference augmented Laplacian: {rel:.3e}"
    )


def test_diag_matches_reference_augmented_laplacian():
    torch.manual_seed(0)
    ei = both_dir_edge_index()
    w = torch.randn(ei.size(1), D) * 0.5
    maps = torch.diag_embed(w)
    x = torch.randn(N, D, F_DIM)

    y_ref = (
        dense_reference_laplacian(ei, maps, augmented=True) @ x.reshape(N * D, F_DIM)
    ).reshape(N, D, F_DIM)

    layer = DiagonalNSDConv(D, F_DIM, 8, context_dim=D * F_DIM)
    layer.W1.data = torch.eye(D)
    layer.W2.data = torch.eye(F_DIM)
    layer.alpha.data = torch.tensor(1.0)
    layer.sigma = nn.Identity()
    pairs = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.size(1))]

    def fake_products(x_feat, ei2):
        E2 = ei2.size(1)
        sm = torch.zeros(E2, D)
        cm = torch.zeros(E2, D)
        for k in range(E2):
            s, t = int(ei2[0, k]), int(ei2[1, k])
            if s == t:
                wt = ws = torch.full((D,), 1e-4)  # real generator: near-zero loop maps
            else:
                wt, ws = w[pairs.index((t, s))], w[pairs.index((s, t))]
            sm[k], cm[k] = wt * wt, wt * ws
        return sm, cm

    layer.get_map_products = fake_products  # ty: ignore[invalid-assignment]
    layer.eval()
    with torch.no_grad():
        out = layer(torch.zeros(N, D * F_DIM), x, ei)
    y_user = x - out
    rel = (y_user - y_ref).norm() / y_ref.norm()
    assert rel < 1e-8, (
        f"rel Frobenius error vs reference augmented Laplacian: {rel:.3e}"
    )


def test_normalization_not_scale_invariant():
    """Augmented normalization must attenuate the Laplacian when maps -> 0.

    On current code the normalized operator is 0-homogeneous in the maps,
    so the gain=0.01 'warm start' does nothing.
    """
    torch.manual_seed(0)
    ei = both_dir_edge_index()
    E = ei.size(1)
    F_maps = torch.randn(E, D, D)
    sm = F_maps.transpose(-2, -1) @ F_maps
    cm = F_maps.transpose(-2, -1) @ torch.randn(E, D, D)

    _, nc_big = apply_general_norm(sm, cm, ei, N, D, training=False)
    _, nc_small = apply_general_norm(sm * 1e-4, cm * 1e-4, ei, N, D, training=False)
    ratio = nc_small.abs().mean() / nc_big.abs().mean()
    assert ratio < 0.01, (
        f"normalized cross-map magnitude ratio small/big maps = {ratio:.3f}; "
        "normalization is scale-invariant (missing +I augmentation)"
    )


def test_train_eval_operator_consistency_at_init():
    """Train-mode jitter must not change the operator by orders of magnitude."""
    torch.manual_seed(3)
    ei = both_dir_edge_index()
    E = ei.size(1)
    F_maps = torch.randn(E, D, D) * 1e-4  # gain=0.01 init scale
    sm = F_maps.transpose(-2, -1) @ F_maps
    cm = F_maps.transpose(-2, -1) @ (torch.randn(E, D, D) * 1e-4)

    _, nc_eval = apply_general_norm(sm, cm, ei, N, D, training=False)
    devs = []
    for seed in range(5):
        torch.manual_seed(seed)
        _, nc_train = apply_general_norm(sm, cm, ei, N, D, training=True)
        devs.append(((nc_train - nc_eval).norm() / (nc_eval.norm() + 1e-30)).item())
    assert max(devs) < 0.1, (
        f"train/eval operator rel deviation at init: {max(devs):.3e}"
    )


def test_dataset_undirected_and_loop_free():
    """Every dataset must reach the model symmetrized and de-looped."""
    from torch_geometric.utils import contains_self_loops, is_undirected

    from exp.data import load_dataset

    for name in ("texas", "cornell", "chameleon"):
        try:
            data, _ = load_dataset(name, root="exp/data")
        except Exception as exc:  # offline CI: skip, do not pass silently
            pytest.skip(f"dataset {name} unavailable: {exc}")
        assert data.edge_index is not None and data.x is not None
        assert is_undirected(data.edge_index), f"{name}: edge_index is directed"
        assert not contains_self_loops(data.edge_index), f"{name}: has self-loops"
        row_sums = data.x.sum(dim=-1)
        ok = ((row_sums - 1.0).abs() < 1e-5) | (row_sums == 0)
        assert bool(ok.all()), f"{name}: features not row-normalized"


def dense_fixed_channel_reference(edge_index, w_maps, add_lp, add_hp):
    """Dense reference operator with fixed lp/hp channels (diagonal variant).

    Learned channels follow the augmented diagonal formula; the fixed
    channels have diagonal deg/(deg+1) and off-diagonal +/-1 scaled by
    (deg_u+1)^{-1/2}(deg_v+1)^{-1/2}, as in the reference builders.
    """
    fd = D + int(add_lp) + int(add_hp)
    E = edge_index.size(1)
    deg = torch.zeros(N)
    for e in range(E):
        deg[int(edge_index[0, e])] += 1
    Ddiag = torch.zeros(N, fd)
    for e in range(E):
        u = int(edge_index[0, e])
        Ddiag[u, :D] += w_maps[e] ** 2
    Ddiag[:, D:] = deg[:, None]
    dinv = (Ddiag + 1).pow(-0.5)

    L = torch.zeros(N * fd, N * fd)
    for u in range(N):
        L[u * fd : (u + 1) * fd, u * fd : (u + 1) * fd] = torch.diag(
            dinv[u] * Ddiag[u] * dinv[u]
        )
    pairs = [(int(edge_index[0, k]), int(edge_index[1, k])) for k in range(E)]
    fixed_offdiag = ([1.0] if add_lp else []) + ([-1.0] if add_hp else [])
    for e in range(E):
        u, v = pairs[e]
        rev = pairs.index((v, u))
        block = torch.zeros(fd)
        block[:D] = -w_maps[e] * w_maps[rev]
        block[D:] = torch.tensor(fixed_offdiag)
        L[u * fd : (u + 1) * fd, v * fd : (v + 1) * fd] = torch.diag(
            dinv[u] * block * dinv[v]
        )
    return L


@pytest.mark.parametrize("add_lp,add_hp", [(True, False), (False, True), (True, True)])
def test_fixed_channels_match_reference_operator(add_lp, add_hp):
    """add_lp/add_hp channels must reproduce the reference fixed blocks."""
    torch.manual_seed(0)
    ei = both_dir_edge_index()
    fd = D + int(add_lp) + int(add_hp)
    w = torch.rand(ei.size(1), D) * 1.6 - 0.8
    x = torch.randn(N, fd, F_DIM)

    y_ref = (
        dense_fixed_channel_reference(ei, w, add_lp, add_hp) @ x.reshape(N * fd, F_DIM)
    ).reshape(N, fd, F_DIM)

    layer = DiagonalNSDConv(D, F_DIM, 8, context_dim=1, add_lp=add_lp, add_hp=add_hp)
    layer.W1.data = torch.eye(fd)
    layer.W2.data = torch.eye(F_DIM)
    layer.alpha.data = torch.tensor(1.0)
    layer.sigma = nn.Identity()
    pairs = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.size(1))]
    w_dst = torch.stack([w[pairs.index((t, s))] for s, t in pairs])
    layer.get_map_products = (  # ty: ignore[invalid-assignment]
        lambda xf, e2: layer._append_fixed_diag(w_dst**2, w_dst * w)
    )
    layer.eval()
    with torch.no_grad():
        out = layer(torch.zeros(N, 1), x, ei)
    rel = (x - out - y_ref).norm() / y_ref.norm()
    assert rel < 1e-10, f"fixed-channel operator rel err {rel:.3e}"


def test_sparse_learner_matches_reference_reduction():
    """sparse_learner must implement the LocalConcatSheafLearnerVariant input:
    view the concatenated pair as [final_d, 2f], sum the stalk axis, linear+tanh.
    """
    torch.manual_seed(0)
    ei = both_dir_edge_index()
    layer = DiagonalNSDConv(D, F_DIM, 8, context_dim=D * F_DIM, sparse_learner=True)
    learner = layer.map_generator[0]  # ty: ignore[not-subscriptable]
    assert learner.in_features == 2 * F_DIM

    x_feat = torch.randn(N, D * F_DIM)
    w_dst, w_src = layer._bidirectional_input(x_feat, ei)

    src, dst = ei
    lin = layer.map_generator[0]  # ty: ignore[not-subscriptable]
    expected_src = torch.tanh(
        lin(
            torch.cat([x_feat[src], x_feat[dst]], dim=-1)
            .view(-1, D, 2 * F_DIM)
            .sum(dim=1)
        )
    )
    assert torch.allclose(w_src, expected_src, atol=1e-12)


def test_second_linear_and_final_d_shapes():
    """second_linear adds an encoder stage; final_d sizes every component."""
    from sheaf_mpnn.nsd import NSDModel, NSDVariant

    torch.manual_seed(0)
    m = NSDModel(
        10,
        3,
        stalk_dim=2,
        hidden_dim=8,
        num_layers=2,
        variant=NSDVariant.GENERAL,
        add_lp=True,
        add_hp=True,
        second_linear=True,
        sparse_learner=True,
    )
    assert m.encoder2 is not None
    assert m.final_d == 4
    assert m.encoder.out_features == 4 * 8
    assert m.decoder.in_features == 4 * 8
    x = torch.randn(6, 10)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    out = m(x, ei)
    assert out.shape == (6, 3) and torch.isfinite(out).all()


def test_reference_options_across_families():
    """add_lp/add_hp/sparse_learner/second_linear apply to every undirected
    family; the shared fixed-channel and sparse-input machinery lives in
    BaseSheafConv, so one structural check per family suffices.
    """
    from sheaf_mpnn.nsd import NSDModel, NSDVariant
    from sheaf_mpnn.nsd.poly_nsd_model import PolyNSDModel, PolyNSDVariant
    from sheaf_mpnn.nsp import NSPModel, NSPVariant
    from sheaf_mpnn.sheaf_adj import SheafAdjModel, SheafAdjVariant

    torch.manual_seed(0)
    x = torch.randn(6, 10)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    kw = dict(add_lp=True, add_hp=True, sparse_learner=True, second_linear=True)

    # Heterogeneous (model class, variant enum) pairs; typed as Any so the
    # shared call site below checks against no single constructor signature.
    families: list[tuple[Any, Any]] = [
        (NSDModel, NSDVariant.GENERAL),
        (NSPModel, NSPVariant.GENERAL),
        (SheafAdjModel, SheafAdjVariant.GENERAL),
        (PolyNSDModel, PolyNSDVariant.GENERAL),
    ]
    for cls, variant in families:
        m = cls(10, 3, stalk_dim=2, hidden_dim=8, num_layers=2, variant=variant, **kw)
        layer = m.layers[0]
        assert m.final_d == 4 and layer.stalk_dim == 4 and layer.d == 2
        assert layer.W1.shape == (4, 4)
        # Sparse learner input is 2f, independent of final_d.
        learner = layer.map_generator[0]
        in_features = learner.in_features
        assert in_features == 2 * 8
        assert m.encoder2 is not None
        out = m(x, ei)
        assert out.shape == (6, 3) and torch.isfinite(out).all()


def test_edge_weighted_bundle_matches_reference_operator():
    """use_edge_weights must reproduce the reference weighted bundle: learned
    channels use the weighted degree sum(w^2), fixed lp/hp channels keep the
    combinatorial degree, both (D+1)-augmented.
    """
    from sheaf_mpnn.nsd.nsd_layers import OrthogonalNSDConv

    torch.manual_seed(0)
    ei = both_dir_edge_index()
    E = ei.size(1)
    pairs = [(int(ei[0, k]), int(ei[1, k])) for k in range(E)]
    rev = [pairs.index((t, s)) for s, t in pairs]
    fd = D + 2  # add_lp + add_hp

    # Random rotations (d=2) and symmetric weights.
    theta = torch.rand(E) * 6.28
    rot = torch.stack(
        [
            torch.stack([theta.cos(), -theta.sin()], -1),
            torch.stack([theta.sin(), theta.cos()], -1),
        ],
        -2,
    )
    w = torch.rand(E, 1) * 0.8 + 0.1
    w2 = (w * w[rev]).squeeze(-1) ** 2

    deg = torch.zeros(N)
    wdeg = torch.zeros(N)
    for k in range(E):
        deg[int(ei[0, k])] += 1
        wdeg[int(ei[0, k])] += w2[k]

    # Dense reference: weighted learned block + combinatorial fixed channels.
    L = torch.zeros(N * fd, N * fd)
    dinv_w, dinv_c = (wdeg + 1).pow(-0.5), (deg + 1).pow(-0.5)
    for u in range(N):
        blk = torch.zeros(fd, fd)
        blk[:D, :D] = (wdeg[u] / (wdeg[u] + 1)) * torch.eye(D)
        blk[D, D] = blk[D + 1, D + 1] = deg[u] / (deg[u] + 1)
        L[u * fd : (u + 1) * fd, u * fd : (u + 1) * fd] = blk
    for k in range(E):
        src, dst = pairs[k]
        blk = torch.zeros(fd, fd)
        # Row dst, column src: the message aggregates -cross[k] at dst, with
        # cross[k] = O_dst^T O_src.
        blk[:D, :D] = -w2[k] * dinv_w[dst] * dinv_w[src] * (rot[rev[k]].T @ rot[k])
        blk[D, D] = dinv_c[dst] * dinv_c[src]  # lp: +1 off-diagonal
        blk[D + 1, D + 1] = -dinv_c[dst] * dinv_c[src]  # hp: -1 off-diagonal
        L[dst * fd : (dst + 1) * fd, src * fd : (src + 1) * fd] = blk

    x = torch.randn(N, fd, F_DIM)
    y_ref = (L @ x.reshape(N * fd, F_DIM)).reshape(N, fd, F_DIM)

    layer = OrthogonalNSDConv(
        D,
        F_DIM,
        8,
        context_dim=1,
        add_lp=True,
        add_hp=True,
        use_edge_weights=True,
    )
    layer.W1.data = torch.eye(fd)
    layer.W2.data = torch.eye(F_DIM)
    layer.alpha.data = torch.tensor(1.0)
    layer.sigma = nn.Identity()

    def products(xf, e2):
        cross = rot[rev].transpose(-2, -1) @ rot
        _, cross = layer._append_fixed_block(cross, w2[:, None, None] * cross)
        self_diag = cross.new_ones(E, fd)
        self_diag[:, :D] = w2.unsqueeze(-1)
        return self_diag, cross

    layer.get_map_products = products  # ty: ignore[invalid-assignment]
    layer.eval()
    with torch.no_grad():
        out = layer(torch.zeros(N, 1), x, ei)
    rel = (x - out - y_ref).norm() / y_ref.norm()
    assert rel < 1e-10, f"weighted bundle rel err {rel:.3e}"


def test_learn_alpha_flag():
    """learn_alpha=False must pin alpha at its init value (Bodnar-exact update)
    through real optimizer steps, while everything else keeps training.
    """
    from sheaf_mpnn.nsd import NSDModel, NSDVariant

    torch.manual_seed(0)
    x = torch.randn(6, 10)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    y = torch.tensor([0, 1, 2, 0, 1, 2])

    m = NSDModel(
        10,
        3,
        stalk_dim=2,
        hidden_dim=8,
        num_layers=2,
        variant=NSDVariant.GENERAL,
        learn_alpha=False,
    )
    assert all(not layer.alpha.requires_grad for layer in m.layers)
    for layer in m.layers:
        assert layer.alpha.item() == 1.0  # ty: ignore[call-non-callable]

    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    enc_before = m.encoder.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x, ei), y).backward()
        opt.step()
    for layer in m.layers:
        assert layer.alpha.item() == 1.0  # ty: ignore[call-non-callable]
    assert not torch.allclose(m.encoder.weight, enc_before)

    # Default stays learnable.
    m2 = NSDModel(10, 3, stalk_dim=2, hidden_dim=8, num_layers=1)
    assert m2.layers[0].alpha.requires_grad
