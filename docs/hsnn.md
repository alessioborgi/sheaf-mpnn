# Undirected Hypergraph Sheaf Neural Networks

`HSNNModel` extends the NSD package with real, unit-weight, undirected hypergraph
message passing. It follows the existing encoder, layer stack, decoder and variant
factory structure. The core is a separate `BaseHypergraphSheafConv`: graph
`BaseSheafConv` assumes pairs of nodes, whereas hypergraph incidences connect two
independent index spaces. Existing graph methods and normalization are unchanged.

See also the [Chinese implementation and review guide](hsnn-implementation-zh.md).

## Python API

```python
import torch
from sheaf_mpnn import HSNNModel, HSNNVariant

# Three nodes in hyperedge 0; nodes 1 and 3 in hyperedge 1.
# Node 4 is isolated and hyperedge 2 is empty.
index = torch.tensor([[0, 1, 2, 1, 3], [0, 0, 0, 1, 1]])
x = torch.randn(5, 8)
model = HSNNModel(
    in_channels=8,
    out_channels=3,
    stalk_dim=4,
    hidden_dim=16,
    num_layers=2,
    variant=HSNNVariant.DIAGONAL,
)
logits = model(x, index, num_hyperedges=3)  # [5, 3], no softmax
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0, 1]))
loss.backward()
```

The public layers are `DiagonalHSNNConv`, `GeneralHSNNConv`,
`OrthogonalHSNNConv` and `LowRankHSNNConv`. All are available from `sheaf_mpnn`
and `sheaf_mpnn.nsd`. Their shared calling convention is:

```python
layer(x_feat, x_stalk, hyperedge_index, num_hyperedges=M)
```

`x_feat` has shape `[N, context_dim]`, stalks have shape `[N, d, f]`, and
`hyperedge_index` is `torch.long[2, K]`. Row 0 contains node IDs and row 1
hyperedge IDs. Matching numeric IDs are valid incidences, not self-loops.
Features and layer parameters must share device and dtype (float32 or float64).
The model handles a single full-batch hypergraph with at least one node.

Provide `num_hyperedges` to retain trailing empty hyperedges; otherwise it is
inferred as `max(edge_ids) + 1`, or zero for an empty index. Unordered incidences,
isolated nodes, singleton hyperedges and empty hyperedges are supported. Repeated
copies of the same incidence are rejected. Different hyperedge IDs may have
identical member sets. Do not apply graph self-loop removal or `to_undirected`
to this index. PyG's default graph batching offsets are not a hypergraph batching
contract; minibatching is outside this initial implementation.

## Restriction maps and propagation

Each layer forms the mean node context of a hyperedge and concatenates it with
each incident node's context. A bias-free linear predictor with Tanh produces
one restriction map per incidence. Every layer predicts fresh maps from its
current stalk features; maps and degrees stay in the autograd graph.

| Variant | Map | Predictor output per incidence |
| --- | --- | --- |
| `DIAGONAL` | Diagonal entries, stored as `[K, d]` | `d` |
| `GENERAL` | Full matrix, stored as `[K, d, d]` | `d*d` |
| `ORTHOGONAL` | Existing Cayley/Householder-family construction | Strategy dependent |
| `LOW_RANK` | Pure `A @ B.T`, without an added diagonal | `2*d*rank` |

The orthogonal variant accepts `orth_strategy="cayley"`, `"fasth"`, `"fasthpp"`
or `"householder"`. The low-rank variant requires `1 <= rank <= d`. At `d=1`,
some orthogonal strategies have no trainable map degrees of freedom. Orthogonal
and low-rank maps are structured alternatives, not a strict expressivity ranking.
This pure low-rank construction follows the existing library's parameterization;
it is not a claim of exact reproduction of other HSNN low-rank architectures.

Let the conceptual block incidence matrix have blocks $B_{ev}=F_{ve}$, with
$F_{ve}:\mathbb{R}^d\to\mathbb{R}^d$. Define

$$
D_v=\sum_{e\ni v}F_{ve}^{\top}F_{ve},\qquad
D=\operatorname{blockdiag}(D_v),\qquad
D_E=\operatorname{blockdiag}(|e|I_d).
$$

The unnormalized Laplacian and propagation are

$$
L=D-B^{\top}D_E^{-1}B,\qquad
R=(D^{\dagger})^{1/2},\qquad
P=RB^{\top}D_E^{-1}BR.
$$

Empty hyperedges contribute no messages. The pseudoinverse discards eigenvalues
at or below `max_eigenvalue * d * finfo(dtype).eps` per node block. The diagonal
path uses the same threshold on diagonal degree entries. Zero blocks yield zero
inverse roots. Full-matrix powers reuse the library's symmetric matrix-power
helper with its repeated-singular-value-safe backward. Pseudoinverse derivatives
are meaningful away from numerical rank changes; rank-deficient and repeated
spectra also have explicit finite-gradient tests.

`degree_regularization=tau > 0` instead computes $R=(D+\tau I)^{-1/2}$ with
the same numerical cutoff. No identity augmentation is silently enabled.
For singular degrees, $RDR$ is a support projection, so
$RLR=RDR-P$, not generally $I-P$. With positive regularization, $RDR$ also
need not equal $I$.

The layer update is

$$
Z=W_1\operatorname{Dropout}(X)W_2,\qquad
X^+=\operatorname{ELU}(PZ+rX).
$$

`residual=False` means $r=0$ (the default), and `True` means $r=1$.
`W1` starts as identity and `W2` is orthogonally initialized. The encoder uses
Linear, ELU and dropout, as in the existing model organization. Dropout inside
each convolution affects only the propagation input; the residual is un-dropped.
An isolated node has zero pure propagation, with behavior after activation and
decoding determined by the residual setting and decoder bias.

`apply_diffusion(x_stalk, index, maps, num_hyperedges=M)` exposes just $PX$ for
fixed-map validation; it omits $W_1/W_2$, dropout, activation and residual.
`get_restriction_maps(x_feat, index, num_hyperedges=M)` exposes the learned maps.

The two passes transport nodes into mean hyperedge states using $F_{ve}$ and
return states using the **same** $F_{ve}^{\top}$. The implementation never builds
$B$, $L$ or a clique expansion. Transport costs $O(Kdf)$ for diagonal maps and
$O(Kd^2f)$ for full maps, plus map prediction and node-degree inverse roots.
Duplicate-incidence validation uses uniqueness checks, so total runtime also
includes input validation. Each layer forward validates incidences once; standalone
map prediction and propagation still validate their own inputs. No validation
results or learned maps are cached across calls. Diagonal degrees need $O(Nd)$ storage; full degrees
need $O(Nd^2)$ storage and their factorization costs $O(Nd^3)$.

The layer-level `hidden_dim` is retained for call compatibility with the initial
NSD-style API; it does not control the single-linear map predictor. Its input
width is `2 * context_dim` and its output width follows the map family. In contrast,
`HSNNModel.hidden_dim` controls the actual number of feature channels per stalk.

## Relation to the notebook and reference implementations

The mathematical starting point is repository notebook
[`8_Hypergraph_Sheaf_Neural_Networks.ipynb`](https://github.com/alessioborgi/sheaf-mpnn/blob/main/notebooks/8_Hypergraph_Sheaf_Neural_Networks.ipynb),
especially its unnormalized operator and dense examples in sections 3.2 and 5.1.
The notebook's section 7 is a teaching scaffold with unimplemented transport
methods. This library fills those roles using incidence message passing.

The corrected operator is also consistent with the undirected ($q=0$) setting of
[Directional Sheaf Hypergraph Networks, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/83ccb398f3ce9c4d137011f36a03c7d4-Paper-Conference.pdf),
section 3 and appendix E, and its
[official repository](https://github.com/EmaMule/DirectionalSheafHypergraphs).
Reference code was inspected at commit
[`d94f81bc7d86bd8d41e416083063f3446c1ad796`](https://github.com/EmaMule/DirectionalSheafHypergraphs/tree/d94f81bc7d86bd8d41e416083063f3446c1ad796).
The implementation here is written in this library's style, without copying the
external sparse-Laplacian construction or detaching its map learner.

Three choices are explicit:

- $L_{vv}=\sum_{e\ni v}(1-1/|e|)F_{ve}^{\top}F_{ve}$. The old diagonal
  sign-flip construction in the reference's `real_sheaf_layers.py` is not used.
- Normalization uses $D_v=\sum F^{\top}F$, not the notebook section 6
  demonstration's `diag(L)` and not graph NSD's augmented `D+I` wrappers.
- This model uses propagation with ELU and an optional input residual. The
  notebook's section 7 and existing graph NSD residual diffusion update
  `X - alpha * ELU(LZ)` are different architectures. There is no claim that
  their complete network outputs are identical.

With identity maps, the propagation operator recovers symmetric HGNN
normalization. For size-two hyperedges the unnormalized $L$ is **one half** of
the usual graph sheaf Laplacian with the same endpoint maps. A singleton has
zero unnormalized Laplacian contribution but can still change normalization
through its contribution to $D$.

## Validation and scope

```bash
uv run pytest tests/test_hypergraph_sheaf_conv.py tests/nsd/test_hsnn_layers.py tests/nsd/test_hsnn_model.py --no-cov
uv run python scripts/smoke_hsnn.py --variant all --steps 60 --seed 42
```

Tests compare transport and input/map gradients to independent block-matrix
calculations, verify energy and spectral properties, and cover permutations,
singular degrees, empty incidences, serialization and short optimization.
The smoke script prints JSON losses and checks finite, nonzero gradients and
parameter updates. Its synthetic loss reduction is not benchmark evidence.

The `hsnn_cora` preset now registers HSNN with the unified `sheaf run` CLI,
using a separate audited hypergraph protocol and frozen Cora split bank.
Directed hypergraphs, learned hyperedge weights, heterogeneous mirrors, static
sheaf sharing, alternative hyperedge context predictors,
benchmark comparisons and ablations are subsequent work. Reproducing a reference
experiment requires matching those additional architectural and data choices.

## Validation snapshot: 2026-11-09

After the snapshot below, input validation was consolidated to run once per
layer forward while retaining validation on both standalone APIs. Seven new
forward/gradient equivalence cases and the HSNN operator plus complete NSD test
suite passed (335 tests). Ruff, changed-file formatting and repository-wide ty
also passed. The full-suite and wheel results below, and the original review
archive, predate this refinement; they were not regenerated for this change.

The configured project `.venv` was used on Windows with Python 3.13.18,
PyTorch 2.8.0+cu128, PyG 2.7.0, NumPy 2.4.3, Ruff 0.15.8 and ty 0.0.42.
CUDA checks ran on an NVIDIA GeForce GTX 1650. No further packages were installed
for this verification.

| Check | Result |
| --- | --- |
| HSNN operator, layer and model tests | 117 passed; includes 14 CPU/CUDA output and gradient parity cases with zero and positive degree regularization |
| Complete non-integration repository suite | 4,379 passed, 7 skipped, 760 integration cases deselected |
| Repository statement coverage | 94% |
| `ruff check .` and `ruff format --check .` | Passed |
| `ty check .` against the configured `.venv` | Passed |

Six repository skips are conditional cases without applicable parameters or
identity shortcuts. One graph-reference parity test could not download Cornell
under the network sandbox. None of the HSNN cases were skipped. The existing
compiled `torch_householder` extension initially emitted a Windows DLL load
diagnostic, so an earlier validation run used the library's existing native
FastH++ fallback for the Householder strategy. A subsequent local repair rebuilt
the unchanged upstream C++ source against the active PyTorch 2.8.0+cu128 with
MSVC 19.51.36260. The compiled backend then passed 12 CPU/CUDA forward/gradient
comparisons (float32 and float64, stalk dimensions 1, 3 and 5) and 332 orthogonal,
NSD and HSNN regression cases with fallback warnings treated as errors. The
final focused and full-suite results above were rerun after that repair, also
treating fallback warnings as errors.

The built wheel was checked for the HSNN modules and for exclusion of build
caches and environments. All four variants passed forward/backward checks when
imported from the unpacked wheel outside the editable source tree.

The smoke script used seed 42, 60 Adam steps and the same synthetic hypergraph
on each device. Every parameter received finite, nonzero gradients and was
updated in every run.

| Variant | CPU initial loss | CPU final loss | CUDA initial loss | CUDA final loss |
| --- | ---: | ---: | ---: | ---: |
| Diagonal | 0.636526 | 0.012972 | 0.636526 | 0.012972 |
| General | 0.822537 | 0.020069 | 0.822537 | 0.020069 |
| Orthogonal | 0.637407 | 0.011920 | 0.637407 | 0.011920 |
| Low-rank | 0.637573 | 0.007639 | 0.639194 | 0.008053 |

These are implementation smoke results, not held-out benchmark comparisons.
Floating-point differences can affect the numerical rank cutoff and training
trajectory, especially for the unregularized low-rank variant.

To reproduce the focused checks from PowerShell in this checkout:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_hypergraph_sheaf_conv.py tests/nsd/test_hsnn_layers.py tests/nsd/test_hsnn_model.py --no-cov
.\.venv\Scripts\python.exe scripts/smoke_hsnn.py --variant all --steps 60 --seed 42 --device cpu
.\.venv\Scripts\python.exe scripts/smoke_hsnn.py --variant all --steps 60 --seed 42 --device cuda
```

UTF-8 mode is also needed for older repository tests that read UTF-8 YAML files
without an explicit encoding on Windows. Temporary test directories were placed
under this checkout's ignored `tmp/` directory.

## Audited Cora experiment

The `sheaf run --preset hsnn_cora` runner uses the pinned Directional Sheaf Hypergraph Networks
loader's topology and a frozen bank of ten splits. The earlier conversion from
`hypergraph.pickle` is superseded: it did not match that loader's topology.
See [data provenance, protocol and reproduction limits](hsnn-cora-protocol-zh.md).
This is an HSNN experiment on the reference topology, not a full DSHN reproduction.

## Optional Householder extension on Windows

A working MSVC compiler and an installed extension can still have a PyTorch C++
ABI mismatch. In the local repair above, the old `.pyd` requested eight symbols
that the active PyTorch's `c10.dll` did not export; the C++ runtime DLL imports
were present. Rebuilding the same source against the active environment removed
all missing imports. PyTorch and the sheaf algorithms were kept unchanged.

The upstream [build configuration](https://github.com/toshas/torch-householder/blob/master/pyproject.toml)
accepts `torch>=1.5`. An isolated build can therefore select a different PyTorch
from the runtime environment. When rebuilding this extension, activate an **x64
Native Tools Command Prompt** and use the active environment's PyTorch. For a
future clean reinstall, the following commands install the build prerequisites
and rebuild just the optional package:

```bat
uv pip install --python .venv\Scripts\python.exe setuptools wheel ninja
uv pip install --python .venv\Scripts\python.exe --no-build-isolation --no-binary torch-householder --reinstall-package torch-householder --no-cache --no-deps torch-householder==1.0.1
```

`--no-build-isolation` selects the already-installed build dependencies, and
`--no-deps` preserves the installed PyTorch. Keep uv's cache and temporary paths
on the desired drive through `UV_CACHE_DIR`, `TEMP` and `TMP`. A rebuild is needed
again if the runtime's PyTorch/Python ABI changes.

The local repair stored the old binary and its installation record under
`tmp/householder-torch28/backup/`, then atomically replaced only the installed
extension and updated its `RECORD` hash. Build products remained in the project's
ignored `tmp/` directory. The source `.cpp` was not edited.

Restart running notebook kernels after changing the extension: a process can
retain its previously loaded DLL or cache the earlier fallback decision. In a
fresh process, verify that the compiled backend is selected:

```python
import torch
import torch_householder_cpp
from sheaf_mpnn.utils.orthogonal import _load_torch_householder_orgqr

assert _load_torch_householder_orgqr() is not None
print(torch.__version__)
print(torch_householder_cpp.__file__)
```
