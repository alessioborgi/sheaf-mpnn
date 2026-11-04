# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import pytest
import torch
from torch_geometric.utils import degree

from sheaf_mpnn.utils.connection_laplacian import (
    TransportMapCache,
    _ensure_min_degree,
    _local_pca,
    _neighbourhood_matrix,
    _orthogonal_alignment,
    generate_connection_laplacian,
)


def test_ensure_min_degree_raises_when_stalk_dim_too_large():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="stalk_dim"):
        _ensure_min_degree(x, edge_index, stalk_dim=3)


def test_ensure_min_degree_no_change_when_sufficient():
    x = torch.randn(4, 8)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [1, 2, 3, 0, 2, 3, 0, 1],
        ],
        dtype=torch.long,
    )
    result = _ensure_min_degree(x, edge_index, stalk_dim=2)
    assert result.shape == edge_index.shape


def test_ensure_min_degree_adds_edges_for_isolated_node():
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    result = _ensure_min_degree(x, edge_index, stalk_dim=3)
    deg = degree(result[0], num_nodes=5)
    assert (deg >= 3).all()


def test_neighbourhood_matrix_shape():
    num_nodes, feat_dim = 5, 8
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 3],
            [1, 2, 0, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    nbhd_mat = _neighbourhood_matrix(x, edge_index)
    max_degree = int(degree(edge_index[0], num_nodes=num_nodes).max().item())
    assert nbhd_mat.shape == (num_nodes, feat_dim, max_degree)


def test_neighbourhood_matrix_centred_values():
    x = torch.arange(6, dtype=torch.float).view(3, 2)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    nbhd_mat = _neighbourhood_matrix(x, edge_index)
    assert nbhd_mat.shape[0] == 3
    expected_node0 = x[1] - x[0]
    assert torch.allclose(nbhd_mat[0, :, 0], expected_node0)


def test_neighbourhood_matrix_unused_slots_zero():
    num_nodes, feat_dim = 4, 4
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    nbhd_mat = _neighbourhood_matrix(x, edge_index)
    assert (nbhd_mat[3] == 0).all()


def test_local_pca_shape():
    num_nodes, feat_dim, stalk_dim = 6, 8, 3
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            [1, 2, 3, 0, 2, 4, 0, 1, 5, 0, 4, 5, 1, 3, 5, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    node_bases = _local_pca(x, edge_index, stalk_dim=stalk_dim)
    assert node_bases.shape == (num_nodes, feat_dim, stalk_dim)


def test_local_pca_orthonormal_columns():
    num_nodes, feat_dim, stalk_dim = 5, 6, 2
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1],
        ],
        dtype=torch.long,
    )
    node_bases = _local_pca(x, edge_index, stalk_dim=stalk_dim)
    gram = torch.bmm(node_bases.transpose(1, 2), node_bases)
    eye = torch.eye(stalk_dim).unsqueeze(0).expand(num_nodes, -1, -1)
    assert torch.allclose(gram, eye, atol=1e-5)


def test_orthogonal_alignment_shape():
    num_nodes, feat_dim, stalk_dim = 5, 8, 3
    node_bases = torch.linalg.qr(torch.randn(num_nodes, feat_dim, stalk_dim))[0]
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    maps = _orthogonal_alignment(node_bases, edge_index)
    assert maps.shape == (3, stalk_dim, stalk_dim)


def test_orthogonal_alignment_orthogonal():
    num_nodes, feat_dim, stalk_dim = 6, 8, 3
    node_bases = torch.linalg.qr(torch.randn(num_nodes, feat_dim, stalk_dim))[0]
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    maps = _orthogonal_alignment(node_bases, edge_index)
    num_edges = maps.shape[0]
    eye = torch.eye(stalk_dim).unsqueeze(0).expand(num_edges, -1, -1)
    assert torch.allclose(maps.transpose(1, 2) @ maps, eye, atol=1e-5)


def test_orthogonal_alignment_identity_for_equal_bases():
    num_nodes, feat_dim, stalk_dim = 4, 6, 3
    node_bases = torch.linalg.qr(torch.randn(1, feat_dim, stalk_dim))[0].expand(
        num_nodes, -1, -1
    )
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    maps = _orthogonal_alignment(node_bases, edge_index)
    eye = torch.eye(stalk_dim).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(maps, eye, atol=1e-5)


def test_generate_connection_laplacian_output_shape():
    num_nodes, feat_dim, stalk_dim = 8, 10, 3
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ],
        dtype=torch.long,
    )
    maps = generate_connection_laplacian(x, edge_index, stalk_dim=stalk_dim)
    assert maps.dim() == 3
    assert maps.shape[1] == stalk_dim
    assert maps.shape[2] == stalk_dim


def test_generate_connection_laplacian_maps_orthogonal():
    num_nodes, feat_dim, stalk_dim = 8, 10, 2
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ],
        dtype=torch.long,
    )
    maps = generate_connection_laplacian(x, edge_index, stalk_dim=stalk_dim)
    eye = torch.eye(stalk_dim).unsqueeze(0).expand(maps.shape[0], -1, -1)
    assert torch.allclose(maps.transpose(1, 2) @ maps, eye, atol=1e-5)


def test_generate_connection_laplacian_symmetrises_edges():
    num_nodes, feat_dim, stalk_dim = 6, 8, 2
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    maps = generate_connection_laplacian(x, edge_index, stalk_dim=stalk_dim)
    assert maps.shape[0] % 2 == 0


def test_local_pca_padding_invariance():
    """Zero-padding to max degree must not change the tangent-space subspace.

    B_padded B_padded^T == B_unpadded B_unpadded^T because zero columns
    contribute nothing, so the top-d left singular vectors (and hence the
    projection P = U U^T) must be identical for all nodes.
    """
    torch.manual_seed(0)
    num_nodes, feat_dim, stalk_dim = 6, 8, 2
    x = torch.randn(num_nodes, feat_dim)
    # Deliberately unequal degrees: node 0 has 4 neighbours, others have 2.
    # Every node has at least stalk_dim=2 neighbours so unpadded SVD is valid.
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [1, 2, 3, 4, 0, 2, 0, 3, 0, 4, 0, 5, 0, 1],
        ],
        dtype=torch.long,
    )

    bases_padded = _local_pca(x, edge_index, stalk_dim)

    src, dst = edge_index[0], edge_index[1]
    bases_unpadded = torch.zeros(num_nodes, feat_dim, stalk_dim)
    for i in range(num_nodes):
        nbrs = dst[src == i]
        nbhd = (x[nbrs] - x[i]).T  # [feat_dim, deg_i]
        U, _, _ = torch.linalg.svd(nbhd, full_matrices=False)
        bases_unpadded[i] = U[:, :stalk_dim]

    # Compare via projection matrices P = U U^T (sign-invariant).
    P_padded = torch.bmm(bases_padded, bases_padded.transpose(1, 2))
    P_unpadded = torch.bmm(bases_unpadded, bases_unpadded.transpose(1, 2))
    torch.testing.assert_close(P_padded, P_unpadded, atol=1e-5, rtol=1e-5)


def test_orthogonal_alignment_antisymmetry():
    """For an undirected graph, O_{ij} = O_{ji}^T.

    Follows from the Procrustes solution: SVD(A^T) = V Sigma U^T, so the
    nearest orthogonal matrix to A^T is V U^T = (U V^T)^T.
    """
    torch.manual_seed(0)
    num_nodes, feat_dim, stalk_dim = 5, 6, 2
    node_bases = torch.linalg.qr(torch.randn(num_nodes, feat_dim, stalk_dim))[0]
    # Symmetric edge set: both (i, j) and (j, i) present.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )

    maps = _orthogonal_alignment(node_bases, edge_index)

    src, dst = edge_index[0], edge_index[1]
    for e in range(edge_index.shape[1]):
        i, j = src[e].item(), dst[e].item()
        (rev_e,) = ((src == j) & (dst == i)).nonzero(as_tuple=True)
        torch.testing.assert_close(maps[e], maps[rev_e[0]].T, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("stalk_dim", [1, 2, 4])
def test_generate_connection_laplacian_various_stalk_dims(stalk_dim):
    num_nodes, feat_dim = 10, 12
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ],
        dtype=torch.long,
    )
    maps = generate_connection_laplacian(x, edge_index, stalk_dim=stalk_dim)
    assert maps.shape[1:] == (stalk_dim, stalk_dim)


class TestTransportMapCache:
    """Tests for TransportMapCache invalidation and memory-bound semantics."""

    @pytest.fixture()
    def graph(self):
        """Bidirectional ring: every node has out-degree 2, satisfying stalk_dim=2."""
        torch.manual_seed(0)
        n, p = 8, 10
        x = torch.randn(n, p)
        idx = torch.arange(n)
        edge_index = torch.stack(
            [torch.cat([idx, idx]), torch.cat([idx.roll(-1), idx.roll(1)])]
        )
        return x, edge_index

    def test_first_call_populates_cache(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        assert cache._maps is None
        cache.get(x, edge_index, stalk_dim=2)
        assert cache._maps is not None

    def test_hit_returns_same_object(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        maps1 = cache.get(x, edge_index, stalk_dim=2)
        maps2 = cache.get(x, edge_index, stalk_dim=2)
        assert maps1 is maps2

    def test_hit_on_same_values_new_tensor(self, graph):
        """A fresh tensor with identical edge_index values must still hit the cache.

        In the model forward(), to_undirected() always returns a new tensor object,
        so pointer-based keying would miss on every call for a fixed graph.
        """
        x, edge_index = graph
        cache = TransportMapCache()
        maps1 = cache.get(x, edge_index, stalk_dim=2)
        maps2 = cache.get(x, edge_index.clone(), stalk_dim=2)
        assert maps1 is maps2

    def test_miss_on_new_x_tensor(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        maps1 = cache.get(x, edge_index, stalk_dim=2)
        maps2 = cache.get(torch.randn_like(x), edge_index, stalk_dim=2)
        assert maps1 is not maps2

    def test_miss_on_inplace_x_write(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        maps1 = cache.get(x, edge_index, stalk_dim=2)
        x.add_(1.0)  # increments x._version
        maps2 = cache.get(x, edge_index, stalk_dim=2)
        assert maps1 is not maps2

    def test_miss_on_changed_edge_index(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        maps1 = cache.get(x, edge_index, stalk_dim=2)
        # Shift neighbours by one step — still a valid bidirectional ring.
        edge_index2 = edge_index.roll(2, dims=1)
        maps2 = cache.get(x, edge_index2, stalk_dim=2)
        assert maps1 is not maps2

    def test_store_bounded_to_one_entry(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        cache.get(x, edge_index, stalk_dim=2)
        cache.get(torch.randn_like(x), edge_index, stalk_dim=2)
        assert cache._maps is not None
        assert cache._edge_index is not None

    def test_x_key_stable_for_same_tensor(self, graph):
        x, _ = graph
        assert TransportMapCache._x_key_for(x) == TransportMapCache._x_key_for(x)

    def test_x_key_differs_for_different_tensor(self, graph):
        x, _ = graph
        assert TransportMapCache._x_key_for(x) != TransportMapCache._x_key_for(
            torch.randn_like(x)
        )

    def test_x_key_differs_after_inplace_write(self, graph):
        x, _ = graph
        key_before = TransportMapCache._x_key_for(x)
        x.add_(1.0)
        assert TransportMapCache._x_key_for(x) != key_before


class TestTransportMapCachePrepare:
    """Tests for ``TransportMapCache.prepare``, which caches edge preprocessing.

    The preprocessing is a pure function of its inputs but runs a near-full KNN
    search whenever ``stalk_dim`` exceeds the mean degree, so it must not be
    repeated on every forward pass.
    """

    @pytest.fixture()
    def graph(self):
        """Sparse ring: mean degree 2, so stalk_dim=4 forces degree augmentation."""
        torch.manual_seed(0)
        n, p = 12, 6
        x = torch.randn(n, p)
        idx = torch.arange(n)
        return x, torch.stack([idx, idx.roll(-1)])

    def test_prepare_augments_degree_and_aligns_maps(self, graph):
        x, edge_index = graph
        prepared, maps = TransportMapCache().prepare(x, edge_index, 4, False)
        deg = degree(prepared[0], num_nodes=x.size(0))
        assert (deg >= 4).all()
        assert maps.shape == (prepared.size(1), 4, 4)

    def test_prepare_adds_self_loops_before_computing_maps(self, graph):
        x, edge_index = graph
        no_loops, maps_no = TransportMapCache().prepare(x, edge_index, 4, False)
        loops, maps_yes = TransportMapCache().prepare(x, edge_index, 4, True)
        assert loops.size(1) == no_loops.size(1) + x.size(0)
        # Maps must stay index-aligned with the edge_index the layers receive.
        assert maps_yes.size(0) == loops.size(1)
        assert maps_no.size(0) == no_loops.size(1)

    def test_prepare_hit_returns_same_objects(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        ei1, maps1 = cache.prepare(x, edge_index, 4, False)
        ei2, maps2 = cache.prepare(x, edge_index.clone(), 4, False)
        assert ei1 is ei2
        assert maps1 is maps2

    def test_prepare_miss_on_new_x(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        _, maps1 = cache.prepare(x, edge_index, 4, False)
        _, maps2 = cache.prepare(torch.randn_like(x), edge_index, 4, False)
        assert maps1 is not maps2

    def test_prepare_miss_on_changed_self_loop_flag(self, graph):
        x, edge_index = graph
        cache = TransportMapCache()
        ei1, _ = cache.prepare(x, edge_index, 4, False)
        ei2, _ = cache.prepare(x, edge_index, 4, True)
        assert ei1 is not ei2

    def test_prepare_and_get_do_not_share_entries(self, graph):
        """``get`` takes a prepared edge_index, ``prepare`` a raw one.

        Both key on the tensor they are handed, so without a per-method marker
        ``get`` could return maps sized for the *augmented* edge set.
        """
        x, edge_index = graph
        cache = TransportMapCache()
        prepared, prep_maps = cache.prepare(x, edge_index, 4, False)
        get_maps = cache.get(x, prepared, 4)
        assert get_maps.size(0) == prepared.size(1)
        # prepare() must recompute rather than hand back get()'s entry.
        _, prep_maps2 = cache.prepare(x, edge_index, 4, False)
        assert prep_maps2.size(0) == prepared.size(1)
        assert prep_maps2 is not get_maps

    def test_prepare_matches_uncached_computation(self, graph):
        x, edge_index = graph
        prepared, maps = TransportMapCache().prepare(x, edge_index, 4, False)
        expected = generate_connection_laplacian(x, edge_index, 4)
        assert maps.shape == expected.shape
        torch.testing.assert_close(maps, expected, atol=1e-5, rtol=1e-5)


def test_prepare_result_survives_inference_mode_then_training():
    """Entries cached during a validation pass must still work for training.

    Lightning runs a sanity-check validation loop before training, so the first
    call lands inside ``torch.inference_mode``. Tensors created there cannot be
    saved for backward, so caching them raw would break the first training step
    with "Inference tensors cannot be saved for backward".
    """
    torch.manual_seed(0)
    n, p, d = 10, 6, 3
    x = torch.randn(n, p)
    idx = torch.arange(n)
    edge_index = torch.stack([idx, idx.roll(-1)])
    cache = TransportMapCache()

    with torch.inference_mode():  # the sanity-check pass populates the cache
        cache.prepare(x, edge_index, d, False)

    prepared, maps = cache.prepare(x, edge_index, d, False)
    assert not prepared.is_inference()
    assert not maps.is_inference()

    # The real check: indexing with the cached edge_index must be differentiable.
    weights = torch.randn(n, d, requires_grad=True)
    (weights[prepared[0]] * maps[:, :, 0]).sum().backward()
    assert weights.grad is not None
