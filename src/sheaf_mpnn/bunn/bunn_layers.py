# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

r"""The Bundle Neural Network (BuNN) convolution layer.

BuNN (Bamberger et al., ICLR 2025) replaces local message passing with
*message diffusion*: one layer evolves node features by the heat equation on a
learned flat vector bundle (Eq. 1-4 of the paper),

.. math::

    O_v = \phi(G, P, X, v), \qquad
    h_v = O_v^\top W O_v x_v + b, \qquad
    Z = \mathcal{H}_\mathcal{B}(t)\,H, \qquad
    X' = \sigma(Z).

Because the bundle is *flat* -- every transport factorises as
:math:`O_{vu} = O_v^\top O_u` -- the bundle heat kernel reduces to the plain
graph heat kernel (Lemma 3.1):

.. math::

    \left(\mathcal{H}_\mathcal{B}(t) X\right)_v
        = \sum_{u} \mathcal{H}(t, v, u)\, O_v^\top O_u x_u,

so a layer is *synchronise* with :math:`O_v`, diffuse scalar channels,
*desynchronise* with :math:`O_v^\top`.  The graph heat kernel itself has no
learned content, which is what makes the whole thing cheap.

The paper evaluates :math:`\mathcal{H}(t) = e^{-t\mathcal{L}}` two ways for
the same layer -- a truncated Taylor series for small :math:`t` (Algorithm 1)
and the exact spectral solution for large :math:`t` (Algorithm 2) -- so that
choice is an argument here, not a separate model.  The nonlinearity
:math:`\sigma` of Eq. 4 lives in
:class:`~sheaf_mpnn.bunn.bunn_model.BuNNModel`, between layers.
"""

import math
from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from sheaf_mpnn.bunn.bundle import BundleMapLearner, PhiKind, desync, sync

BuNNSolver = Literal["taylor", "spectral"]
BUNN_SOLVERS: tuple[str, ...] = ("taylor", "spectral")

Normalization = Literal["sym", "rw"]
NORMALIZATIONS: tuple[str, ...] = ("sym", "rw")

# Largest eigenvalue of either normalised Laplacian (attained on bipartite
# graphs); both spectra lie in [0, 2].
LAMBDA_MAX = 2.0

# The degree-K truncation of exp(-t*lambda) is only usable while K stays ahead
# of t*lambda: the terms (t*lambda)^k / k! peak at k ~ t*lambda with magnitude
# ~ e^{t*lambda}, so a truncation below that peak returns a large positive
# number instead of a small decaying one -- it does not converge slowly, it
# diverges. Measured relative error at lambda = LAMBDA_MAX stays under 1% while
# K >= 4 * t * lambda (t=1/K=8 sits right at that boundary, which is the
# paper's own default pairing), with a floor for the very small times where the
# linear rule would ask for a degree or two.
TAYLOR_SAFETY = 4.0
MIN_TAYLOR_DEGREE = 4

# Past this point *no* degree rescues the series: the partial sums grow to
# ~e^{t*lambda} on the way to an answer of size e^{-t*lambda}, so the result is
# pure cancellation error. Measured in float32, the best achievable relative
# error is ~1e-4 at t*lambda = 8, 0.2 at t*lambda = 10, and 3e4 at
# t*lambda = 16. Above the cap the spectral solver is the only correct option,
# which is exactly the split the paper describes -- truncated Taylor for small
# t, spectral for large t.
TAYLOR_MAX_TAU_LAMBDA = 8.0
TAYLOR_MAX_TAU = TAYLOR_MAX_TAU_LAMBDA / LAMBDA_MAX

# Dense eigendecomposition is O(N^3) time and O(N^2) memory. Past this many
# nodes the [N, N] eigenvector matrix alone runs to gigabytes (ogbn-arxiv's
# 169k nodes would need ~115 GB), so the spectral solver refuses rather than
# thrashing. Appendix A of the paper flags this cost as the framework's main
# practical limitation.
MAX_DENSE_EIGH_NODES = 20_000


def stable_taylor_degree(tau: float) -> int:
    """Smallest Taylor degree that keeps ``exp(-tau * L)`` accurate.

    Only meaningful for ``tau <= TAYLOR_MAX_TAU``; beyond that no degree is
    enough (see :data:`TAYLOR_MAX_TAU_LAMBDA`).

    Args:
        tau: Diffusion time.

    Returns:
        The minimum ``max_degree`` for which the truncation is trustworthy.
    """
    return max(MIN_TAYLOR_DEGREE, math.ceil(TAYLOR_SAFETY * LAMBDA_MAX * tau))


def _sparse_low_frequency_eigenpairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_eigs: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The ``num_eigs`` smallest eigenpairs of the sym-normalised Laplacian.

    Computed by shift-invert Lanczos (``scipy.sparse.linalg.eigsh``) on the
    *sparse* Laplacian, so the dense ``[N, N]`` matrix is never formed and the
    dense-eigh node cap does not apply.  This is the paper's truncated spectral
    solution (Appendix A) for graphs too large for a full decomposition; since
    ``exp(-t lambda)`` suppresses the high-frequency end, the retained
    low-frequency eigenpairs are what the heat kernel keeps at large ``t``.

    The construction mirrors :func:`laplacian_eigendecomposition` exactly --
    same sym-normalised weights, same unit diagonal on isolated nodes, same
    undirected-graph check -- so the two solvers agree wherever both can run.

    Args:
        edge_index: Graph connectivity ``[2, E]``, both directions stored.
        num_nodes: Number of nodes ``N``.
        num_eigs: Number of low-frequency eigenpairs to return; must be ``< N``.
        dtype: Floating dtype of the returned tensors.

    Returns:
        ``(eig_vals, eig_vecs)`` with shapes ``[num_eigs]`` and
        ``[N, num_eigs]``, ascending by eigenvalue, columns orthonormal.

    Raises:
        ValueError: If ``num_eigs >= num_nodes`` (Lanczos needs ``k < N``), or
            if ``edge_index`` is not symmetric.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse import eye as sparse_eye
    from scipy.sparse.linalg import eigsh

    # eigsh's Lanczos needs k < N. The caller has bounded num_eigs to
    # [1, num_nodes]; this path only runs on huge graphs, so a real truncation
    # always satisfies it -- but a full-spectrum request cannot be served here.
    if num_eigs >= num_nodes:
        raise ValueError(
            f"the sparse solver needs num_eigs < num_nodes ({num_nodes}), got "
            f"{num_eigs}; the full spectrum requires the dense solver"
        )

    src, dst = edge_index
    # float64 throughout the decomposition for headroom; cast to dtype at the end.
    deg = degree(dst, num_nodes=num_nodes, dtype=torch.float64)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    weight = (deg_inv_sqrt[src] * deg_inv_sqrt[dst]).cpu().numpy()
    rows, cols = dst.cpu().numpy(), src.cpu().numpy()

    # Duplicate (parallel) edges sum on the csr conversion, matching the dense
    # path's accumulate=True; isolated nodes keep the unit diagonal from I.
    a_norm = coo_matrix((weight, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    lap = sparse_eye(num_nodes, format="csr", dtype=np.float64) - a_norm

    # eigsh assumes symmetry (reads one triangle); a directed edge_index would
    # otherwise return the spectrum of a graph the caller never passed.
    asym = lap - lap.transpose()
    if asym.nnz and abs(asym).max() > 1e-6:
        raise ValueError(
            "the spectral solver requires an undirected graph (both directions "
            "of every edge present), but edge_index is asymmetric; symmetrise "
            "it with torch_geometric.utils.to_undirected, or use solver='taylor'"
        )

    # Shift just below the spectrum ([0, 2], with a genuine 0 eigenvalue): sigma
    # slightly negative keeps L - sigma*I positive definite so the internal
    # factorisation succeeds, and which='LM' then returns the eigenvalues
    # nearest sigma -- the low-frequency end.
    eig_vals_np, eig_vecs_np = eigsh(lap, k=num_eigs, sigma=-1e-3, which="LM")
    order = np.argsort(eig_vals_np)

    device = edge_index.device
    eig_vals = torch.from_numpy(eig_vals_np[order]).to(device=device, dtype=dtype)
    eig_vecs = torch.from_numpy(eig_vecs_np[:, order]).to(device=device, dtype=dtype)
    return eig_vals, eig_vecs


def laplacian_eigendecomposition(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype = torch.float32,
    num_eigs: int | None = None,
    max_dense_nodes: int = MAX_DENSE_EIGH_NODES,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Eigendecomposition of the symmetric normalised graph Laplacian.

    Builds :math:`\mathcal{L} = I - D^{-1/2} A D^{-1/2}` densely and calls
    ``torch.linalg.eigh``.  Isolated nodes carry a unit diagonal -- their signal
    simply decays -- which is also what the Taylor recurrence's
    :math:`I - A_\mathrm{norm}` does, so the two solvers agree on every graph.

    Above ``max_dense_nodes`` the dense path would need a multi-gigabyte
    ``[N, N]`` matrix, so a *truncated* request (``num_eigs`` set) is instead
    routed to :func:`_sparse_low_frequency_eigenpairs`, which returns the same
    low-frequency eigenpairs without ever forming that matrix.  A full-spectrum
    request on such a graph is refused -- there is no cheap way to get it.

    The graph must be undirected.  ``eigh`` reads only one triangle of its
    input and assumes the other mirrors it, so an asymmetric adjacency would
    be silently symmetrised and the returned spectrum would belong to a
    different graph than the one passed in; this function checks instead.

    Args:
        edge_index: Graph connectivity ``[2, E]``, with both directions of
            every edge stored.
        num_nodes: Number of nodes ``N``.
        dtype: Floating dtype of the returned tensors.
        num_eigs: Keep only the ``num_eigs`` smallest eigenpairs (the paper's
            *truncated* spectral solution). Since ``exp(-t lambda)`` suppresses
            the high-frequency end, dropping it is a good approximation for
            large ``t`` -- but it is an approximation: the retained
            eigenvectors no longer span the space, so short diffusions lose
            signal. ``None`` keeps the full spectrum and is exact.
        max_dense_nodes: Above this many nodes the dense ``eigh`` is skipped: a
            truncated request goes to the sparse solver, a full-spectrum one is
            refused (see :data:`MAX_DENSE_EIGH_NODES`).

    Returns:
        ``(eig_vals, eig_vecs)`` with shapes ``[K]`` and ``[N, K]``, ascending
        by eigenvalue; the columns of ``eig_vecs`` are orthonormal.

    Raises:
        ValueError: If the graph is too large and the full spectrum is
            requested, if ``num_eigs`` is out of range, or if ``edge_index`` is
            not symmetric.
    """
    if num_eigs is not None and not 1 <= num_eigs <= num_nodes:
        raise ValueError(f"num_eigs must be in [1, {num_nodes}], got {num_eigs}")

    if num_nodes > max_dense_nodes:
        if num_eigs is None:
            raise ValueError(
                f"the full spectral solver needs a dense {num_nodes}x{num_nodes} "
                f"eigendecomposition (~{num_nodes**2 * 4 / 1e9:.1f} GB), above the "
                f"max_dense_nodes={max_dense_nodes} limit; set num_eigs to keep "
                "only the low-frequency eigenpairs (computed sparsely, no dense "
                "matrix), use solver='taylor', which is sparse, or precompute "
                "eigenvectors and pass them in"
            )
        # Truncated request on a graph too large for a dense eigh: compute just
        # the low-frequency eigenpairs sparsely, which lifts the node cap.
        return _sparse_low_frequency_eigenpairs(edge_index, num_nodes, num_eigs, dtype)

    device = edge_index.device
    src, dst = edge_index
    deg = degree(dst, num_nodes=num_nodes, dtype=dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    lap = torch.eye(num_nodes, dtype=dtype, device=device)
    # accumulate=True so parallel edges sum instead of overwriting each other.
    lap.index_put_((dst, src), -weight, accumulate=True)

    if not torch.allclose(lap, lap.transpose(0, 1), atol=1e-6):
        raise ValueError(
            "the spectral solver requires an undirected graph (both directions "
            "of every edge present), but edge_index is asymmetric; symmetrise "
            "it with torch_geometric.utils.to_undirected, or use "
            "solver='taylor'"
        )

    eig_vals, eig_vecs = torch.linalg.eigh(lap)
    if num_eigs is not None:
        # eigh returns ascending eigenvalues, so the leading columns are the
        # low-frequency end that survives the heat kernel.
        eig_vals, eig_vecs = eig_vals[:num_eigs], eig_vecs[:, :num_eigs]
    return eig_vals, eig_vecs


class BuNNConv(MessagePassing):
    r"""One BuNN layer: bundle maps, bundle-aware encoder, heat diffusion.

    Implements Eq. 1-3 of the paper (the Eq. 4 nonlinearity is applied by the
    model).  The layer state is a flat vector of width ``num_bundles *
    bundle_dim * in_channels`` per node, reshaped internally to
    ``[N, b, d, f]``: Appendix E's several bundles, each carrying several
    vector-field channels.  Using ``b`` low-dimensional bundles instead of one
    ``bd``-dimensional one is the paper's practical choice -- it parameterises
    the block-diagonal subgroup of :math:`O(bd)` and costs only one angle per
    bundle at ``d = 2``.

    Unlike a sheaf layer, no map is ever computed per *edge*: :math:`\phi` runs
    once per node, which is where the scalability over NSD comes from.
    """

    def __init__(
        self,
        num_bundles: int,
        bundle_dim: int,
        in_channels: int,
        tau: float = 1.0,
        learn_tau: bool = False,
        solver: BuNNSolver = "taylor",
        max_degree: int | None = None,
        num_eigs: int | None = None,
        context_dim: int | None = None,
        pe_size: int = 0,
        normalize: Normalization = "sym",
        bias: bool = True,
        phi: PhiKind = "mlp",
        num_phi_layers: int = 2,
        orth_strategy: Literal["cayley", "fasth", "fasthpp", "householder"] = "cayley",
        clamp_val: float = 10.0,
        external_maps: bool = False,
    ):
        r"""Initializes a BuNN convolution layer.

        Args:
            num_bundles: Number of parallel bundles ``b``.
            bundle_dim: Stalk dimension ``d`` of each bundle. ``2`` (the
                paper's choice throughout) uses the direct O(2) angle
                parameterisation.
            in_channels: Vector-field channels ``f`` per bundle. The node state
                width is ``b * d * f``.
            tau: Diffusion time ``t``, the scale at which messages diffuse.
                Larger ``t`` couples more distant nodes; the paper's grid is
                0.1/1/10/100.
            learn_tau: Learn one diffusion time per bundle (kept non-negative
                by a ReLU), as the reference spectral layer does. Under
                ``"taylor"`` the learned time is additionally clamped to the
                largest value ``max_degree`` can represent, since gradient
                steps would otherwise walk it into the divergent regime.
            solver: How :math:`e^{-t\mathcal{L}}` is evaluated -- ``"taylor"``
                (Algorithm 1: ``max_degree`` sparse products, small ``t``) or
                ``"spectral"`` (Algorithm 2: exact, any ``t``, dense
                eigendecomposition).
            max_degree: Taylor truncation degree ``K`` (``"taylor"`` only).
                ``None`` derives the smallest degree that is accurate for
                ``tau`` (:func:`stable_taylor_degree`), which is what you want
                unless you are deliberately studying the truncation: an
                explicit value too small for ``tau`` is rejected rather than
                silently diffusing with a divergent series.
            num_eigs: Truncate the spectral solver to this many low-frequency
                eigenpairs (``"spectral"`` only). ``None`` is exact.
            context_dim: Width of the node context features ``x_feat`` fed to
                the map learner. Defaults to the state width.
            pe_size: Width of the optional positional encoding ``P`` also fed
                to :math:`\phi` (Eq. 1 reads ``phi(G, P, X, v)``). Injective
                encodings are what Theorem 5.3's universality assumes, and
                they keep the maps from degenerating in deep stacks: without
                them :math:`\phi` sees only features it produced itself, which
                on a long diffusion collapses towards a trivial bundle.
            normalize: Which normalised Laplacian drives the diffusion --
                ``"sym"`` (:math:`I - D^{-1/2} A D^{-1/2}`) or ``"rw"``
                (:math:`I - D^{-1} A`). ``"spectral"`` requires ``"sym"``.
            bias: Whether the Eq. 2 encoder carries a bias ``b``.
            phi: Bundle-map learner backbone (``"mlp"`` or ``"sum_gnn"``).
            num_phi_layers: Depth of the map learner.
            orth_strategy: Orthogonal parameterisation for ``bundle_dim != 2``.
            clamp_val: Clamp on the orthogonal-transform parameters.
            external_maps: If ``True``, the layer owns no map learner and
                ``forward`` requires a precomputed ``node_rep`` -- used when
                the model shares one :math:`\phi` across layers.

        Raises:
            ValueError: On non-positive sizes, a negative ``tau`` or
                ``pe_size``, an unknown ``solver``/``normalize``,
                ``"spectral"`` with ``"rw"``, or a ``max_degree`` too small to
                represent ``tau``.
        """
        super().__init__(aggr="add", node_dim=0)
        if num_bundles <= 0:
            raise ValueError("num_bundles must be positive")
        if pe_size < 0:
            raise ValueError("pe_size must be non-negative")
        if bundle_dim <= 0:
            raise ValueError("bundle_dim must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if tau < 0:
            raise ValueError("tau must be non-negative")
        if max_degree is not None and max_degree <= 0:
            raise ValueError("max_degree must be positive")
        if solver not in BUNN_SOLVERS:
            raise ValueError(f"solver must be one of {BUNN_SOLVERS}, got {solver!r}")
        if normalize not in NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}"
            )
        if solver == "spectral" and normalize != "sym":
            raise ValueError(
                "the spectral solver requires normalize='sym': the rw Laplacian "
                "is not symmetric, so its eigenvectors are not orthonormal and "
                "eig_vecs @ eig_vecs^T would not reconstruct the signal"
            )
        # Refuse rather than return a large positive number where a small
        # decaying one belongs -- a failure that otherwise reads as diverged
        # training rather than as a misconfigured solver.
        if solver == "taylor" and tau > TAYLOR_MAX_TAU:
            raise ValueError(
                f"tau={tau} is beyond what the Taylor solver can evaluate "
                f"(tau <= {TAYLOR_MAX_TAU:g}): the partial sums reach "
                f"~e^{{{tau * LAMBDA_MAX:g}}} on the way to an answer of size "
                f"e^-{{{tau * LAMBDA_MAX:g}}}, so the result is cancellation "
                "error at any max_degree. Use solver='spectral', which is "
                "exact at any diffusion time."
            )
        if max_degree is None:
            max_degree = stable_taylor_degree(tau)
        if solver == "taylor" and max_degree < stable_taylor_degree(tau):
            raise ValueError(
                f"max_degree={max_degree} cannot represent exp(-{tau} * L): the "
                f"Taylor truncation diverges unless max_degree >= "
                f"{stable_taylor_degree(tau)}. Raise max_degree, lower tau, or "
                "use solver='spectral', which is exact at any diffusion time."
            )

        self.num_bundles = num_bundles
        self.bundle_dim = bundle_dim
        self.in_channels = in_channels
        self.width = num_bundles * bundle_dim * in_channels
        self.context_dim = context_dim if context_dim is not None else self.width
        self.pe_size = pe_size
        self.solver: BuNNSolver = solver
        self.normalize: Normalization = normalize
        self.max_degree = max_degree
        self.num_eigs = num_eigs
        self.tau_init = tau
        self.learn_tau = learn_tau
        # Largest diffusion time this solver can represent; a learned tau is
        # held at or below it (None for the exact spectral solver, which has
        # no such limit).
        self.tau_cap = (
            min(max_degree / (TAYLOR_SAFETY * LAMBDA_MAX), TAYLOR_MAX_TAU)
            if solver == "taylor"
            else None
        )

        # One diffusion time per bundle, so bundles can operate at different
        # scales of the graph (reference: learned, initialised in U(0.5, 1.5)*t).
        if learn_tau:
            self.tau = nn.Parameter(torch.empty(num_bundles))
        else:
            self.register_buffer("tau", torch.full((num_bundles,), float(tau)))

        self.linear = nn.Linear(self.width, self.width, bias=bias)
        self.map_learner = (
            None
            if external_maps
            else BundleMapLearner(
                context_dim=self.context_dim + pe_size,
                num_bundles=num_bundles,
                bundle_dim=bundle_dim,
                phi=phi,
                num_layers=num_phi_layers,
                orth_strategy=orth_strategy,
                clamp_val=clamp_val,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.linear.reset_parameters()
        if self.map_learner is not None:
            self.map_learner.reset_parameters()
        if self.learn_tau:
            assert isinstance(self.tau, nn.Parameter)
            with torch.no_grad():
                self.tau.uniform_(0.5, 1.5).mul_(self.tau_init)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_bundle(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes flat node states ``[N, width]`` to ``[N, b, d, f]``."""
        if x.dim() != 2 or x.size(1) != self.width:
            raise ValueError(
                f"x must have shape [num_nodes, {self.width}]"
                f" (= num_bundles * bundle_dim * in_channels), got {tuple(x.shape)}"
            )
        return x.view(x.size(0), self.num_bundles, self.bundle_dim, self.in_channels)

    def _node_maps(
        self,
        x_feat: torch.Tensor,
        edge_index: torch.Tensor,
        node_rep: torch.Tensor | None,
        pe: torch.Tensor | None,
    ) -> torch.Tensor:
        """Returns the ``[N, b, d, d]`` bundle maps, learned here or passed in."""
        if node_rep is not None:
            return node_rep
        if self.map_learner is None:
            raise ValueError(
                "layer was built with external_maps=True; pass node_rep explicitly"
            )
        if self.pe_size:
            if pe is None:
                raise ValueError(
                    f"layer was built with pe_size={self.pe_size}; pass pe to forward"
                )
            if pe.size(-1) != self.pe_size:
                raise ValueError(
                    f"pe has width {pe.size(-1)} but pe_size is {self.pe_size}"
                )
            x_feat = torch.cat([x_feat, pe], dim=-1)
        return self.map_learner(x_feat, edge_index)

    def _edge_norm(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Per-edge weights of the normalised adjacency, plus ``D^{-1}``.

        For ``"sym"`` the weights are :math:`d_u^{-1/2} d_v^{-1/2}` and no
        post-scaling is needed; for ``"rw"`` the weights are ones and the
        returned ``deg_inv`` rescales the sum at the destination.
        """
        src, dst = edge_index
        deg = degree(dst, num_nodes=num_nodes, dtype=dtype)
        if self.normalize == "sym":
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
            return deg_inv_sqrt[src] * deg_inv_sqrt[dst], None
        deg_inv = deg.pow(-1.0)
        deg_inv.masked_fill_(deg_inv == float("inf"), 0.0)
        return torch.ones_like(src, dtype=dtype), deg_inv

    def _scaled_tau(self, x: torch.Tensor) -> torch.Tensor:
        """Per-bundle diffusion times, broadcastable over ``[N, b, d, f]``.

        A learned time is kept non-negative, and under the Taylor solver also
        capped at what the truncation can represent -- otherwise a few gradient
        steps can carry it into the divergent regime mid-training.
        """
        tau = self.tau
        if self.learn_tau:
            tau = torch.relu(tau)
            if self.tau_cap is not None:
                tau = tau.clamp(max=self.tau_cap)
        return tau.to(x.dtype)[None, :, None, None]

    # ------------------------------------------------------------------
    # Heat kernel
    # ------------------------------------------------------------------

    def _diffuse_taylor(
        self, h: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        r"""Truncated Taylor heat kernel (Algorithm 1).

        Evaluates :math:`\sum_{k=0}^{K} \frac{(-t\mathcal{L})^k}{k!} h` with
        ``K`` sparse products, building each term from the previous one as
        :math:`p_k = -\frac{t}{k}\mathcal{L}\,p_{k-1}`.
        """
        num_nodes = h.size(0)
        edge_weight, deg_inv = self._edge_norm(edge_index, num_nodes, h.dtype)
        tau = self._scaled_tau(h)

        out = h
        curr = h  # (-t L)^k / k! applied to h, term by term.
        for k in range(1, self.max_degree + 1):
            agg = self.propagate(  # ty: ignore[missing-argument]
                edge_index,  # ty: ignore[invalid-argument-type]
                x=curr,
                edge_weight=edge_weight,
                size=(num_nodes, num_nodes),
            )
            if deg_inv is not None:
                agg = deg_inv[:, None] * agg
            # L curr = curr - A_norm curr, then fold in the -t/k factor.
            step = (curr - agg).view(
                num_nodes, self.num_bundles, self.bundle_dim, self.in_channels
            )
            curr = ((-tau / k) * step).reshape(num_nodes, self.width)
            out = out + curr
        return out

    def _diffuse_spectral(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        eig_vecs: torch.Tensor | None,
        eig_vals: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""Exact spectral heat kernel (Algorithm 2).

        Evaluates :math:`\sum_i e^{-t\lambda_i}\varphi_i \varphi_i^\top h`,
        exact for any diffusion time.  The eigendecomposition is passed in by
        the model (which caches it per graph) or computed here.
        """
        num_nodes = h.size(0)
        if eig_vecs is None or eig_vals is None:
            eig_vals, eig_vecs = laplacian_eigendecomposition(
                edge_index, num_nodes, dtype=h.dtype, num_eigs=self.num_eigs
            )

        proj = eig_vecs.transpose(0, 1) @ h  # [K, width]
        proj = proj.view(-1, self.num_bundles, self.bundle_dim, self.in_channels)
        decay = torch.exp(-eig_vals[:, None, None, None] * self._scaled_tau(h))
        return eig_vecs @ (proj * decay).reshape(-1, self.width)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_feat: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_rep: torch.Tensor | None = None,
        eig_vecs: torch.Tensor | None = None,
        eig_vals: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Applies one BuNN layer: embed, update, diffuse, return to local frames.

        Args:
            x_feat: Node context features ``[N, context_dim]`` driving the map
                learner :math:`\phi`.
            x: Flat node states ``[N, num_bundles * bundle_dim * in_channels]``.
            edge_index: Graph connectivity ``[2, E]``, assumed undirected.
            node_rep: Optional precomputed bundle maps ``[N, b, d, d]``;
                required when the layer was built with ``external_maps=True``.
            eig_vecs: Optional Laplacian eigenvectors ``[N, K]`` (spectral only).
            eig_vals: Optional Laplacian eigenvalues ``[K]`` (spectral only).
            pe: Positional encodings ``[N, pe_size]``, appended to ``x_feat``
                before :math:`\phi`. Required when ``pe_size > 0``.

        Returns:
            Updated flat node states, same shape as ``x``.
        """
        # Validate before the map learner runs, so a mis-shaped state reports
        # itself rather than surfacing as a matmul error inside phi.
        x_bundle = self._to_bundle(x)
        maps = self._node_maps(x_feat, edge_index, node_rep, pe)

        # Sync (Lemma 3.1): into the global frame, where the bundle heat kernel
        # is the plain graph heat kernel.
        #
        # W and b are applied here, between synchronisation and diffusion, as
        # in Algorithm 1 (lines 1-2). Note this is *not* literally Eq. 2: the
        # bias ends up rotated, O_v^T(W O_v x_v + b), where Eq. 2 writes
        # O_v^T W O_v x_v + b. The two agree only when every O_v is the same.
        # Algorithm 1 and the reference implementation disagree here too (the
        # reference adds b after the diffusion instead), so the three differ by
        # where a constant sits relative to sync and the kernel. We follow the
        # paper's pseudocode.
        h = sync(x_bundle, maps).reshape(x.size(0), self.width)
        h = self.linear(h)

        if self.solver == "taylor":
            h = self._diffuse_taylor(h, edge_index)
        else:
            h = self._diffuse_spectral(h, edge_index, eig_vecs, eig_vals)

        return desync(self._to_bundle(h), maps).reshape(x.size(0), self.width)

    def message(  # ty: ignore[invalid-method-override]
        self, x_j: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j


__all__ = [
    "BuNNConv",
    "BuNNSolver",
    "BUNN_SOLVERS",
    "Normalization",
    "NORMALIZATIONS",
    "laplacian_eigendecomposition",
]
