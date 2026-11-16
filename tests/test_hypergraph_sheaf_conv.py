# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Independent matrix and energy checks for hypergraph sheaf transport."""

import pytest
import torch

from sheaf_mpnn import DiagonalHSNNConv, GeneralHSNNConv


def dense_operators(index, maps, num_nodes, num_edges, regularization=0.0):
    """Build B, D, L, R and P using block loops, independent of production code."""
    if maps.ndim == 2:
        maps = torch.diag_embed(maps)
    d = maps.size(-1)
    zero = maps.new_zeros(d, d)
    rows = []
    degrees = [zero.clone() for _ in range(num_nodes)]
    for edge in range(num_edges):
        blocks = []
        for node in range(num_nodes):
            hits = ((index[0] == node) & (index[1] == edge)).nonzero().flatten()
            block = maps[int(hits[0])] if hits.numel() else zero
            blocks.append(block)
            degrees[node] = degrees[node] + block.T @ block
        rows.append(torch.cat(blocks, dim=1))
    b = torch.cat(rows, dim=0) if rows else maps.new_zeros(0, num_nodes * d)
    degree = torch.block_diag(*degrees)
    inverses = []
    for block in degrees:
        matrix = block + regularization * torch.eye(d, dtype=maps.dtype)
        values, vectors = torch.linalg.eigh(matrix)
        keep = values > values.max() * d * torch.finfo(maps.dtype).eps
        safe = torch.where(keep, values, torch.ones_like(values))
        inverse = torch.where(keep, safe.rsqrt(), torch.zeros_like(values))
        inverses.append((vectors * inverse) @ vectors.T)
    r = torch.block_diag(*inverses)
    edge_inverse = maps.new_zeros(num_edges * d)
    for edge in range(num_edges):
        cardinality = int((index[1] == edge).sum())
        if cardinality:
            edge_inverse[edge * d : (edge + 1) * d] = 1.0 / cardinality
    q = b.T @ torch.diag(edge_inverse) @ b
    lap = degree - q
    return b, degree, lap, r, r @ q @ r


@pytest.fixture
def incidence():
    # Singleton, pair, triple, overlapping edges, isolated node and empty edges.
    return torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 1, 2, 2, 2]])


@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_dense_transport_and_gradients(incidence, diagonal, regularization):
    cls = DiagonalHSNNConv if diagonal else GeneralHSNNConv
    layer = cls(3, 2, 4, degree_regularization=regularization).double()
    shape = (6, 3) if diagonal else (6, 3, 3)
    maps = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    x = torch.randn(5, 3, 2, dtype=torch.float64, requires_grad=True)
    actual = layer.apply_diffusion(x, incidence, maps, num_hyperedges=5)
    *_, p = dense_operators(incidence, maps, 5, 5, regularization)
    expected = (p @ x.reshape(15, 2)).reshape_as(x)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
    # The isolated zero block has undefined eigh vector derivatives. Compare
    # oracle gradients on the active subgraph, whose random degrees are SPD.
    active_x = x[:4]
    *_, active_p = dense_operators(incidence, maps, 4, 3, regularization)
    reference = (active_p @ active_x.reshape(12, 2)).reshape_as(active_x)
    probe = torch.randn_like(active_x)
    gradients = torch.autograd.grad((actual[:4] * probe).sum(), (x, maps))
    ref_gradients = torch.autograd.grad((reference * probe).sum(), (x, maps))
    for observed, ref in zip(gradients, ref_gradients, strict=True):
        torch.testing.assert_close(observed, ref, atol=1e-8, rtol=1e-7)


def test_corrected_triple_laplacian_and_hgnn():
    index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    maps = torch.ones(3, 1, dtype=torch.float64)
    _, _, lap, _, p = dense_operators(index, maps, 3, 1)
    expected_p = torch.ones(3, 3, dtype=torch.float64) / 3
    torch.testing.assert_close(lap, torch.eye(3, dtype=torch.float64) - expected_p)
    torch.testing.assert_close(p, expected_p)
    layer = DiagonalHSNNConv(1, 3, 2).double()
    out = layer.apply_diffusion(torch.eye(3, dtype=torch.float64)[:, None], index, maps)
    torch.testing.assert_close(out[:, 0], expected_p)
    # Old diagonal sign-flipping would give diagonal 1/3, not 2/3.
    torch.testing.assert_close(lap.diag(), torch.full((3,), 2 / 3).double())


def test_identity_maps_match_symmetric_hgnn(incidence):
    h = torch.zeros(5, 5, dtype=torch.float64)
    h[incidence[0], incidence[1]] = 1
    node_degree = h.sum(1)
    node_inv = node_degree.clamp_min(1).rsqrt() * (node_degree > 0)
    edge_inv = h.sum(0).clamp_min(1).reciprocal()
    expected = node_inv[:, None] * (h * edge_inv) @ h.T * node_inv[None, :]
    layer = DiagonalHSNNConv(1, 5, 2).double()
    out = layer.apply_diffusion(
        torch.eye(5, dtype=torch.float64)[:, None],
        incidence,
        torch.ones(6, 1, dtype=torch.float64),
        num_hyperedges=5,
    )
    torch.testing.assert_close(out[:, 0], expected)


def test_laplacian_energy_and_spectrum(incidence):
    maps = torch.randn(6, 3, 3, dtype=torch.float64)
    x = torch.randn(5, 3, dtype=torch.float64)
    _, degree, lap, r, p = dense_operators(incidence, maps, 5, 5)
    energy = x.new_zeros(())
    for edge in range(5):
        hits = (incidence[1] == edge).nonzero().flatten().tolist()
        for position, i in enumerate(hits):
            for j in hits[position + 1 :]:
                difference = maps[i] @ x[incidence[0, i]]
                difference = difference - maps[j] @ x[incidence[0, j]]
                energy = energy + difference.square().sum() / len(hits)
    torch.testing.assert_close(x.flatten() @ lap @ x.flatten(), energy)
    torch.testing.assert_close(lap, lap.T)
    assert torch.linalg.eigvalsh(lap).min() >= -1e-10
    assert torch.linalg.eigvalsh(p).min() >= -1e-10
    assert torch.linalg.eigvalsh(p).max() <= 1 + 1e-10
    torch.testing.assert_close(r @ lap @ r, r @ degree @ r - p)


def test_pair_laplacian_is_half_graph_laplacian():
    index = torch.tensor([[0, 1], [0, 0]])
    maps = torch.randn(2, 2, 2, dtype=torch.float64)
    _, _, lap, _, _ = dense_operators(index, maps, 2, 1)
    coboundary = torch.cat((maps[0], -maps[1]), dim=1)
    torch.testing.assert_close(lap, 0.5 * coboundary.T @ coboundary)


@pytest.mark.parametrize("regularization", [0.0, 0.3])
def test_diagonal_full_matrix_agreement(incidence, regularization):
    diagonal = DiagonalHSNNConv(3, 2, 2, degree_regularization=regularization).double()
    general = GeneralHSNNConv(3, 2, 2, degree_regularization=regularization).double()
    maps = torch.randn(6, 3, dtype=torch.float64)
    maps[:, 1] = 0  # Exact singular direction, including the singleton degree.
    x = torch.randn(5, 3, 2, dtype=torch.float64)
    first = diagonal.apply_diffusion(x, incidence, maps, num_hyperedges=7)
    second = general.apply_diffusion(
        x, incidence, torch.diag_embed(maps), num_hyperedges=7
    )
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("diagonal", [True, False])
def test_gradcheck(diagonal):
    cls = DiagonalHSNNConv if diagonal else GeneralHSNNConv
    layer = cls(2, 2, 2, degree_regularization=0.1).double()
    index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    shape = (4, 2) if diagonal else (4, 2, 2)
    maps = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: layer.apply_diffusion(a, index, b), (x, maps)
    )


@pytest.mark.parametrize("kind", ["zero", "rank_deficient", "repeated"])
@pytest.mark.parametrize("diagonal", [True, False])
def test_singular_and_repeated_degree_backward(kind, diagonal):
    cls = DiagonalHSNNConv if diagonal else GeneralHSNNConv
    layer = cls(3, 2, 2).double()
    index = torch.tensor([[0, 1], [0, 0]])
    diagonal_values = {
        "zero": [0, 0, 0],
        "rank_deficient": [1, 0, 0],
        "repeated": [1, 1, 1],
    }
    values = torch.tensor(diagonal_values[kind], dtype=torch.float64).expand(2, -1)
    maps = values.clone() if diagonal else torch.diag_embed(values)
    maps.requires_grad_()
    x = torch.randn(3, 3, 2, dtype=torch.float64, requires_grad=True)
    out = layer.apply_diffusion(x, index, maps)
    out.square().sum().backward()
    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert maps.grad is not None and torch.isfinite(maps.grad).all()
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]))
    if kind == "zero":
        torch.testing.assert_close(out, torch.zeros_like(out))


@pytest.mark.parametrize("num_edges", [None, 0, 7])
@pytest.mark.parametrize("diagonal", [True, False])
def test_empty_incidences(num_edges, diagonal):
    cls = DiagonalHSNNConv if diagonal else GeneralHSNNConv
    layer = cls(2, 3, 2)
    x = torch.randn(4, 2, 3, requires_grad=True)
    out = layer(
        x.flatten(1), x, torch.empty(2, 0, dtype=torch.long), num_hyperedges=num_edges
    )
    torch.testing.assert_close(out, torch.zeros_like(out))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "index,num_edges,error",
    [
        (torch.tensor([[0], [0], [0]]), None, "shape"),
        (torch.tensor([[0.0], [0.0]]), None, "dtype"),
        (torch.tensor([[-1], [0]]), None, "negative"),
        (torch.tensor([[3], [0]]), None, "node index"),
        (torch.tensor([[0], [2]]), 2, "hyperedge index"),
        (torch.tensor([[0, 0], [0, 0]]), None, "duplicate"),
        (torch.empty(2, 0, dtype=torch.long), -1, "nonnegative"),
        (torch.empty(2, 0, dtype=torch.long), 1.5, "integer"),
    ],
)
def test_invalid_incidence(index, num_edges, error):
    layer = DiagonalHSNNConv(2, 2, 2)
    x = torch.randn(3, 2, 2)
    with pytest.raises(ValueError, match=error):
        layer(x.flatten(1), x, index, num_hyperedges=num_edges)


def test_singleton_and_distinct_identical_hyperedges():
    layer = GeneralHSNNConv(2, 2, 2).double()
    x = torch.randn(1, 2, 2, dtype=torch.float64)
    maps = torch.tensor([[[2.0, 0.1], [0.3, 1.0]]], dtype=torch.float64)
    index = torch.tensor([[0], [0]])  # Matching numeric IDs are a valid incidence.
    torch.testing.assert_close(layer.apply_diffusion(x, index, maps), x)
    doubled = torch.tensor([[0, 0], [0, 1]])
    torch.testing.assert_close(
        layer.apply_diffusion(x, doubled, maps.repeat(2, 1, 1)), x
    )


def test_invalid_maps():
    layer = GeneralHSNNConv(2, 2, 2)
    x = torch.randn(1, 2, 2)
    index = torch.tensor([[0], [0]])
    with pytest.raises(ValueError, match="shape"):
        layer.apply_diffusion(x, index, torch.ones(1, 2))
    with pytest.raises(ValueError, match="dtype"):
        layer.apply_diffusion(x, index, torch.ones(1, 2, 2).double())
    with pytest.raises(ValueError, match="finite"):
        layer.apply_diffusion(x, index, torch.full((1, 2, 2), float("nan")))
