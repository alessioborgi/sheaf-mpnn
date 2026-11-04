# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops, degree, scatter, to_undirected

# Peak size of a single padded neighbourhood / gathered-basis block. Chunking
# to this budget keeps memory independent of the graph's maximum degree. Peak
# usage is a small multiple of this, since the float64 Gram matrix is live at
# the same time as the block it came from.
_CHUNK_BUDGET_BYTES = 128 * 1024 * 1024

# Hard cap on the number of matrices in one batched eigendecomposition.
# cuSOLVER's batched symmetric solver allocates a per-matrix workspace that is
# essentially independent of the matrix size (~0.6 MiB each in float64), so a
# batch of 10k tiny 11x11 Gram matrices asks for ~6 GiB while the matrices
# themselves are under 10 MiB. Only the batch count bounds it. The decomposition
# takes tens of milliseconds either way, so splitting it costs nothing.
_EIGH_MAX_BATCH = 512


def _ensure_min_degree(
    x: Tensor,
    edge_index: Tensor,
    stalk_dim: int,
) -> Tensor:
    """Supplement any node with fewer than ``stalk_dim`` neighbours.

    Adds KNN edges (in Euclidean feature space) only for nodes whose out-degree
    is below ``stalk_dim``, so that local PCA has enough points to span a
    ``stalk_dim``-dimensional subspace.

    Args:
        x: Node feature matrix of shape ``[N, p]``.
        edge_index: Graph connectivity of shape ``[2, E]``.
        stalk_dim: Minimum required out-degree per node.

    Returns:
        Augmented ``edge_index`` of shape ``[2, E']`` with ``E' >= E``.

    Raises:
        ValueError: If ``stalk_dim >= num_nodes`` (not enough nodes to form
            a neighbourhood of the required size).
    """
    if stalk_dim <= 0:
        raise ValueError(f"stalk_dim must be positive, got {stalk_dim}")
    num_nodes = x.size(0)
    if num_nodes <= stalk_dim:
        raise ValueError(
            f"stalk_dim={stalk_dim} requires at least {stalk_dim + 1} nodes, "
            f"but the graph has {num_nodes}."
        )

    deg = degree(edge_index[0], num_nodes=num_nodes)
    under = (deg < stalk_dim).nonzero(as_tuple=True)[0]
    if under.numel() == 0:
        return edge_index

    # Compute distances only for under-connected nodes to avoid O(N^2) work.
    dist2 = torch.cdist(x[under], x)  # [|under|, N]
    dist2[torch.arange(under.numel(), device=x.device), under] = float("inf")
    knn_dst = dist2.topk(stalk_dim, largest=False).indices  # [|under|, stalk_dim]
    knn_src = under.unsqueeze(1).expand(-1, stalk_dim)
    knn_ei = torch.stack([knn_src.reshape(-1), knn_dst.reshape(-1)])
    return to_undirected(torch.cat([edge_index, knn_ei], dim=1), num_nodes=num_nodes)


def _csr_layout(
    src: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sort edges by source node and return a CSR-style layout.

    Args:
        src: Source node of every directed edge, shape ``[E]``.
        num_nodes: Number of nodes ``N``.

    Returns:
        Tuple ``(order, start, deg)`` where ``order`` sorts the edges by source
        (stably), ``start`` has shape ``[N + 1]`` and holds the offset of each
        node's edge block inside the sorted arrays, and ``deg`` is the
        out-degree per node, shape ``[N]``.
    """
    order = src.argsort(stable=True)
    deg = scatter(
        torch.ones(src.size(0), dtype=torch.long, device=src.device),
        src,
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    start = torch.zeros(num_nodes + 1, dtype=torch.long, device=src.device)
    start[1:] = deg.cumsum(0)
    return order, start, deg


def _neighbourhood_matrix(
    x: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Build a zero-padded centred neighbourhood matrix of shape ``[N, p, K]``.

    ``K`` is the maximum out-degree. Column ``B[i, :, k]`` is the centred
    feature vector of node ``i``'s ``k``-th neighbour; unused slots are zero.

    Note:
        This materialises the full ``[N, p, K]`` block and is therefore only
        suitable for small graphs; ``_local_pca`` uses the chunked
        ``_neighbourhood_block`` instead.

    Args:
        x: Node feature matrix of shape ``[N, p]``.
        edge_index: Graph connectivity of shape ``[2, E]``.

    Returns:
        Neighbourhood matrix ``B`` of shape ``[N, p, K]``.
    """
    num_nodes = x.size(0)
    src, dst = edge_index[0], edge_index[1]
    order, start, deg = _csr_layout(src, num_nodes)
    max_degree = int(deg.max().item())
    nodes = torch.arange(num_nodes, device=x.device)
    # Zero-padding does not affect the left singular vectors: B_padded B_padded^T
    # == B_unpadded B_unpadded^T because zero columns contribute nothing.
    return _neighbourhood_block(x, dst[order], start, deg, nodes, max_degree)


def _neighbourhood_block(
    x: Tensor,
    dst_sorted: Tensor,
    start: Tensor,
    deg: Tensor,
    nodes: Tensor,
    num_slots: int,
) -> Tensor:
    """Build the padded neighbourhood matrix for a subset of nodes.

    Args:
        x: Node feature matrix of shape ``[N, p]``.
        dst_sorted: Edge destinations ordered by source, shape ``[E]``.
        start: CSR offsets of shape ``[N + 1]``.
        deg: Out-degree per node, shape ``[N]``.
        nodes: Node ids to include, shape ``[c]``.
        num_slots: Column count ``K`` to pad each neighbourhood to; must be at
            least ``deg[nodes].max()``.

    Returns:
        Neighbourhood matrix of shape ``[c, p, K]``.
    """
    feat_dim = x.size(1)
    lens = deg[nodes]
    rows = torch.repeat_interleave(torch.arange(nodes.numel(), device=x.device), lens)
    # Position of each edge inside its own node's block.
    block_start = lens.cumsum(0) - lens
    slots = torch.arange(int(lens.sum().item()), device=x.device) - block_start[rows]
    edges = start[nodes][rows] + slots

    block = torch.zeros(
        nodes.numel(), num_slots, feat_dim, dtype=x.dtype, device=x.device
    )
    block[rows, slots] = x[dst_sorted[edges]] - x[nodes][rows]
    return block.transpose(1, 2)


def _degree_sorted_chunks(
    deg_sorted: list[int],
    feat_dim: int,
    itemsize: int,
    budget_bytes: int,
    max_rows: int = _EIGH_MAX_BATCH,
) -> list[int]:
    """Split degree-sorted nodes into chunks under a memory budget.

    A chunk of ``c`` nodes padded to ``K`` columns costs ``c * p * K``
    elements. Because the nodes are sorted by degree, ``K`` is the degree of
    the chunk's last node, so grouping similar degrees together keeps the
    padding waste (and hence peak memory) bounded.

    Chunks are additionally capped at ``max_rows`` nodes, which bounds the
    batched-``eigh`` workspace (see ``_EIGH_MAX_BATCH``) independently of how
    small the per-node matrices are.

    Args:
        deg_sorted: Out-degrees in non-decreasing order.
        feat_dim: Feature dimension ``p``.
        itemsize: Bytes per element of the neighbourhood matrix.
        budget_bytes: Target peak size of a single padded block.
        max_rows: Hard cap on the number of nodes per chunk.

    Returns:
        Chunk boundaries ``[0, b_1, ..., N]``, always at least one node wide.
    """
    max_elems = max(1, budget_bytes // (feat_dim * itemsize))
    bounds = [0]
    num_nodes = len(deg_sorted)
    lo = 0
    while lo < num_nodes:
        # (b - lo) * deg_sorted[b - 1] is non-decreasing in b, so binary search
        # for the widest chunk that still fits the budget (at least one node).
        best = lo + 1
        left, right = lo + 1, min(lo + max_rows, num_nodes)
        while left <= right:
            mid = (left + right) // 2
            if (mid - lo) * max(deg_sorted[mid - 1], 1) <= max_elems:
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        lo = best
        bounds.append(lo)
    return bounds


def _block_bases(block: Tensor, stalk_dim: int) -> Tensor:
    """Return the top-``d`` left singular vectors of each matrix in a batch.

    Uses an eigendecomposition of the smaller Gram matrix rather than an SVD of
    ``block`` itself. The padded neighbourhood matrices are strongly
    rank-deficient (typically ``degree << p``, plus zero padding), and
    cuSOLVER's batched Jacobi SVD fails to converge on them, silently falling
    back to a path that is roughly two orders of magnitude slower. ``eigh`` on
    the SPD Gram has no such failure mode.

    Args:
        block: Batch of neighbourhood matrices, shape ``[c, p, K]``.
        stalk_dim: Number of leading left singular vectors to return.

    Returns:
        Orthonormal bases of shape ``[c, p, d]``.
    """
    num_cols = block.size(2)
    # Forming a Gram matrix squares the condition number, so TF32's 10-bit
    # mantissa is not enough here; force full fp32 for these two matmuls.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if num_cols > block.size(1):
            # Wide block (hub nodes): B B^T is the smaller side and its
            # eigenvectors are the left singular vectors directly.
            gram = (block @ block.transpose(1, 2)).double()
            eigvecs = torch.linalg.eigh(gram)[1]
            # eigh returns ascending eigenvalues; flip to descending.
            return eigvecs.flip(-1)[:, :, :stalk_dim].to(block.dtype)

        # Tall block (the common case): eigendecompose the small K x K Gram and
        # push the eigenvectors through B to recover the left singular vectors.
        gram = (block.transpose(1, 2) @ block).double()
        eigvecs = torch.linalg.eigh(gram)[1].flip(-1)[:, :, :stalk_dim]
        scaled = block @ eigvecs.to(block.dtype)  # [c, p, d], columns sigma_i u_i
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # Normalising by sigma would blow up on rank-deficient neighbourhoods, so
    # orthonormalise instead: QR leaves well-determined directions untouched
    # (the columns are already mutually orthogonal) and supplies an arbitrary
    # orthonormal completion where sigma ~ 0, exactly as a thin SVD would.
    return torch.linalg.qr(scaled)[0]


def _local_pca(
    x: Tensor,
    edge_index: Tensor,
    stalk_dim: int,
    budget_bytes: int = _CHUNK_BUDGET_BYTES,
) -> Tensor:
    """Compute per-node orthonormal tangent-space bases via local PCA.

    Nodes are grouped into degree-sorted chunks and each chunk is padded only
    to its own maximum degree, so a handful of hub nodes cannot force the whole
    ``[N, p, max_degree]`` block into memory. Zero-padding leaves ``B B^T``
    unchanged, so the bases do not depend on the chunking.

    Args:
        x: Node feature matrix of shape ``[N, p]``.
        edge_index: Graph connectivity of shape ``[2, E]``.
        stalk_dim: Number of principal components to keep (stalk dimension ``d``).
        budget_bytes: Target peak size of a single padded neighbourhood block.

    Returns:
        Orthonormal bases of shape ``[N, p, d]``.
    """
    num_nodes, feat_dim = x.shape
    src, dst = edge_index[0], edge_index[1]
    order, start, deg = _csr_layout(src, num_nodes)
    dst_sorted = dst[order]

    # Ascending degree, so each chunk pads to a degree close to its own nodes'.
    node_order = deg.argsort(stable=True)
    deg_sorted = deg[node_order].tolist()
    bounds = _degree_sorted_chunks(deg_sorted, feat_dim, x.element_size(), budget_bytes)

    node_bases = torch.empty(
        num_nodes, feat_dim, stalk_dim, dtype=x.dtype, device=x.device
    )
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        nodes = node_order[lo:hi]
        # Pad to at least stalk_dim columns so there are always enough
        # directions to return, even for low-degree chunks.
        num_slots = max(deg_sorted[hi - 1], stalk_dim)
        block = _neighbourhood_block(x, dst_sorted, start, deg, nodes, num_slots)
        node_bases[nodes] = _block_bases(block, stalk_dim)
    return node_bases


def _orthogonal_alignment(
    node_bases: Tensor,
    edge_index: Tensor,
    budget_bytes: int = _CHUNK_BUDGET_BYTES,
) -> Tensor:
    """Compute orthogonal transport maps ``O_ij = U V^T`` for each edge.

    For each directed edge ``(i, j)``, decomposes ``O_i^T O_j`` via SVD and
    returns ``U V^T``, the nearest orthogonal matrix in Frobenius norm. Edges
    are processed in chunks so the gathered ``[E, p, d]`` bases never have to
    be materialised at once.

    Args:
        node_bases: Per-node orthonormal bases of shape ``[N, p, d]``.
        edge_index: Graph connectivity of shape ``[2, E]``.
        budget_bytes: Target peak size of a single gathered basis block.

    Returns:
        Orthogonal transport maps of shape ``[E, d, d]``.
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = src.numel()
    _, feat_dim, stalk_dim = node_bases.shape
    per_edge = feat_dim * stalk_dim * node_bases.element_size()
    chunk = max(1, budget_bytes // per_edge)

    maps = torch.empty(
        num_edges, stalk_dim, stalk_dim, dtype=node_bases.dtype, device=src.device
    )
    for lo in range(0, num_edges, chunk):
        hi = min(lo + chunk, num_edges)
        alignment_mat = torch.bmm(
            node_bases[src[lo:hi]].transpose(1, 2), node_bases[dst[lo:hi]]
        )
        left_vecs, _, right_vecs_h = torch.linalg.svd(
            alignment_mat, full_matrices=False
        )
        maps[lo:hi] = torch.bmm(left_vecs, right_vecs_h)
    return maps


class TransportMapCache:
    """Single-entry cache for per-edge orthogonal transport maps.

    Invalidation strategy
    ---------------------
    ``x`` is identified in O(1) by a ``(data_ptr, shape, _version)`` tuple.
    ``data_ptr`` and ``shape`` detect a new tensor; ``_version`` (PyTorch's
    internal write counter) detects an in-place mutation to the same storage.
    Inference tensors have no such counter and take a constant -1 instead.

    ``edge_index`` is compared by *value* via ``torch.equal``, not by pointer,
    because callers such as ``to_undirected`` return a fresh tensor object on
    every call even for the same graph — pointer identity would always miss.
    A shape pre-check short-circuits ``torch.equal`` when the topology changes.

    Usage::

        cache = TransportMapCache()
        edge_index, maps = cache.prepare(x, edge_index, stalk_dim, self_loops)
    """

    def __init__(self) -> None:
        self._maps: torch.Tensor | None = None
        self._edge_index: torch.Tensor | None = None
        self._prepared: torch.Tensor | None = None
        # Identifies which method produced the entry as well as its inputs, so
        # get() and prepare() cannot hand back each other's cached maps.
        self._key: tuple = ()

    @staticmethod
    def _x_key_for(x: torch.Tensor) -> tuple:
        """Return the O(1) cache key for a node-feature tensor."""
        # Inference tensors (Lightning eval runs under inference_mode) keep no
        # version counter, so reading _version on them raises.
        version = -1 if x.is_inference() else x._version
        return (x.data_ptr(), x.shape, version)

    def get(
        self, x: torch.Tensor, edge_index: torch.Tensor, stalk_dim: int
    ) -> torch.Tensor:
        """Return cached maps, recomputing if ``x`` or ``edge_index`` changed.

        Args:
            x: Node feature matrix ``[N, p]``.
            edge_index: Symmetrised, degree-augmented graph connectivity ``[2, E]``.
            stalk_dim: Tangent-space dimension used to compute the maps.

        Returns:
            Orthogonal transport maps ``[E, d, d]``.
        """
        key = ("get", self._x_key_for(x), stalk_dim)
        maps = self._maps
        if maps is None or self._stale(key, edge_index):
            with torch.inference_mode(False), torch.no_grad():
                node_bases = _local_pca(x, edge_index, stalk_dim)
                maps = _orthogonal_alignment(node_bases, edge_index)
            self._maps = maps
            self._prepared = None
            self._edge_index = edge_index
            self._key = key
        return maps

    def _stale(self, key: tuple, edge_index: torch.Tensor) -> bool:
        """Return whether the cached entry must be recomputed for ``key``."""
        return (
            key != self._key
            or self._edge_index is None
            or edge_index.shape != self._edge_index.shape
            or not torch.equal(edge_index, self._edge_index)
        )

    def prepare(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        stalk_dim: int,
        with_self_loops: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the preprocessed ``edge_index`` and its transport maps.

        Symmetrises, degree-augments and optionally self-loops the raw graph,
        then computes the maps — caching *both* against the raw inputs. The
        preprocessing is a pure function of ``(x, edge_index, stalk_dim,
        with_self_loops)`` but is far from free: when ``stalk_dim`` exceeds the
        mean degree, ``_ensure_min_degree`` runs a KNN search over nearly every
        node, which dominates both runtime and peak memory if repeated on every
        forward pass.

        Args:
            x: Raw node feature matrix ``[N, p]``.
            edge_index: Raw graph connectivity ``[2, E]``.
            stalk_dim: Tangent-space dimension ``d``.
            with_self_loops: Whether to add self-loops. Must happen before the
                maps are computed so that the maps stay index-aligned with the
                ``edge_index`` the layers see.

        Returns:
            Tuple ``(prepared_edge_index, maps)`` of shapes ``[2, E']`` and
            ``[E', d, d]``.
        """
        key = ("prepare", self._x_key_for(x), stalk_dim, with_self_loops)
        prepared, maps = self._prepared, self._maps
        if prepared is None or maps is None or self._stale(key, edge_index):
            num_nodes = x.size(0)
            # inference_mode(False) matters: the first call typically comes from
            # a validation pass (Lightning's sanity check runs before training),
            # and tensors created under inference_mode cannot later be saved for
            # backward. Cached entries outlive that pass, so they must be normal
            # tensors -- an inference-mode edge_index would blow up the first
            # time a training forward indexes with it.
            with torch.inference_mode(False), torch.no_grad():
                prepared = to_undirected(edge_index, num_nodes=num_nodes)
                prepared = _ensure_min_degree(x, prepared, stalk_dim)
                if with_self_loops:
                    prepared, _ = add_self_loops(prepared, num_nodes=num_nodes)
                node_bases = _local_pca(x, prepared, stalk_dim)
                maps = _orthogonal_alignment(node_bases, prepared)
            self._prepared, self._maps = prepared, maps
            # Keyed on the *raw* edge_index, since that is what callers pass.
            self._edge_index = edge_index
            self._key = key
        return prepared, maps


@torch.no_grad()
def generate_connection_laplacian(
    x: Tensor,
    edge_index: Tensor,
    stalk_dim: int,
) -> Tensor:
    """Compute orthogonal connection maps for the connection Laplacian.

    For each node ``i``, SVDs the centred neighbourhood matrix to obtain an
    orthonormal tangent-space basis ``O_i``. For each directed edge ``(i, j)``,
    returns the nearest orthogonal map ``O_i^T O_j ~ U V^T`` that transports
    ``T_{x_i}M`` to ``T_{x_j}M``.

    Args:
        x: Node feature matrix of shape ``[N, p]``.
        edge_index: Graph connectivity of shape ``[2, E]``.
        stalk_dim: Tangent-space / stalk dimension ``d``.

    Returns:
        Orthogonal transport maps of shape ``[E', d, d]`` where ``E'`` is the
        number of directed edges after symmetrisation and degree augmentation.
    """
    num_nodes = x.size(0)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = _ensure_min_degree(x, edge_index, stalk_dim)

    node_bases = _local_pca(x, edge_index, stalk_dim)
    return _orthogonal_alignment(node_bases, edge_index)
