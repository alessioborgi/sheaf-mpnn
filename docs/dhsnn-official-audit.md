# DSHN alignment report — 2026-11-15

Reference: EmaMule/DirectionalSheafHypergraphs commit
d94f81bc7d86bd8d41e416083063f3446c1ad796. The existing DHSNN model was updated;
no parallel legacy model was added. Data/frozen splits were not changed.
This report supersedes the earlier partial audit.

## Implemented correspondence

| Component | Official source | Current implementation |
| --- | --- | --- |
| Input/projector | sheafgedi.py, complex_linear.py | Shared bias-free projection, detached imaginary input |
| Context/learner | directed_sheaf_builders.py | Stalk means, mean-hyperedge projection, var1/var2/var3, configurable activation |
| Map lifetime | SheafGeDi.forward | Static once per forward or independent dynamic learners |
| Transport | directed_sheaf_layers.py | Same factored operator via two incidence passes |
| Normalization | complex_norm.py, general normalise | Four modes, expanded full-map edge degree, identity, detached general block inverse |
| Feature transforms | RealLinear, ComplexLayerNorm | Optional stalk projection and complex batch whitening |
| Update/readout | SheafGeDi.forward, mlp.py | Complex bias/residual, interlayer activation/dropout, MLP, sum before edge classifier |
| Optimizer | main.py:377 | Single Adam LR/weight decay; parsed upstream sheaf_wd is unused |
| Selection | main.py | First strictly best validation checkpoint, configurable patience |

## Independent evidence

The reference integration test executes unmodified official model, builder,
normalization, convolution, classifier and sparse autograd code, not a dense
substitute. Cached source SHA-256 values are recorded and checked. Only an
unused compiled Householder import is stubbed; calling that extension fails.
Official orthogonal network comparisons use Cayley.

Verified configurations:
- 36 float32 forward cases: three families, static/dynamic, three prediction
  methods, whitening off/on, with node and hyperedge readout.
- Twenty-four float64 parameter-gradient and three-step Adam comparisons,
  covering static/dynamic sheaves, node/hyperedge tasks and whitening off/on.
  Across these cases the maximum absolute parameter-gradient error was 8.30e-7.
- Twelve float64 normalization cases: all four modes, three families, q=0.17,
  left projection disabled.
- Input-gradient comparison and CPU/CUDA training-step checks.
- Independent fixed-map dense/layout tests remain separate.

Recorded float32 node-logit error <=5.91e-6, hyperedge-logit error <=9.54e-7.
Double-network tolerance is 2e-6 because upstream creates some count/phase
intermediates in float32. Whitening can amplify roundoff; no bitwise identity
is claimed. Raw results: tmp/dshn-reference-results.xml.

## Real-data training

Local GPU: GeForce GTX 1650, PyTorch 2.8.0+cu128. Fresh single-split runs,
maximum 1500 epochs, patience 200, no hyperparameter search, test disabled:

| Dataset/config | Actual / best epoch | Validation | Elapsed | Peak GPU allocation |
| --- | --- | --- | --- | --- |
| Cora diagonal, d=4,f=16,2 layers | 219 / 19 | accuracy 84.34% | 15.74 s | 85.68 MiB |
| Molecular-2 general, d=4,f=16,2 layers | 692 / 492 | F1 90.09% | 53.26 s | 101.22 MiB |

Both two-epoch CLI checks passed. These validation values cannot be compared
directly with paper ten-split test means or the previous test result.
Artifacts: tmp/dshn-alignment-*/.

## Performance

A warmed general-model microbenchmark, equal 251 parameters, 256 nodes,
128 edges, 1024 incidences, d=2,f=3, ten forward/backward steps:

| Device | Message passing | Official sparse assembly |
| --- | --- | --- |
| CPU, two threads | 31.20 ms/step | 45.47 ms/step |
| GTX 1650 | 51.17 ms/step | 73.21 ms/step |

CUDA peak allocations were about 18.11 and 22.51 MiB; both models were resident.
These are local microbenchmark measurements, not a universal speed guarantee.
Static map prediction is reused per forward and global matrix assembly is
avoided. Raw artifact: tmp/dshn-performance.json.

## Boundaries

### Further implementation update

Random hyperedge initialization is now available as dhsnn_init_hedge=rand.
The random raw features are nonpersistent buffers reused across forwards and
cleared on reset. Matrix exponential maps are available through
dhsnn_orthogonal_map=matrix_exp. Euler is deliberately not exposed: although
the upstream utility implements it, the DSHN builder rejects it.
The expanded reference suite covers average/random initialization, full/Light,
directed/undirected roles and native Cayley/matrix exponential configurations.
Random contexts are synchronized explicitly before comparison. In double
precision this avoids the upstream random initializer's hardcoded complex64
dtype mismatch; this is a portability correction, not an accuracy claim.

### Additional Light and undirected verification

`ModelConfig.dhsnn_light` now selects factor-detached message passing. Actual
torch_sparse 0.6.18 sparse products were checked to return detached values.
Twelve direct upstream comparisons cover full/Light, directed/all-positive
undirected incidence roles, and diagonal/Cayley/general maps. Forward outputs
and parameter gradients (including missing learner gradients) passed. These
checks establish the tested undirected operator convention, not reproduction
of every published undirected dataset experiment. The runner allows missing
gradients only for Light map learners; other gradient checks remain active.

Remaining work includes random hyperedge initialization, other upstream
orthogonal parameterizations, the standalone HSNN baseline whole-network
comparison, official search configuration extraction and full paired training.
The following historical boundaries should be read with this update: Light
is now implemented and directly tested, but large-scale Light results are not
yet available.

- Equivalence covers tested configurations, not every numerical edge case.
- Compiled upstream Householder, matrix-exp, random hyperedge initialization
  and Light are not claimed as reproduced. Native Householder/FastH and pure
  low rank are supported library extensions.
- Public-loader default splits are verified; private paper-run RNG states
  are unavailable. Model seed streams remain independent from split RNG.
- Width, depth, epochs, patience and 250-trial search spaces remain explicit
  library experimental choices, not verified paper-selected settings.
- Old checkpoints and search signatures are incompatible; use fresh outputs.
- No full 112-combination rerun or large sweep was started. No server files or
  prior experiment outputs were changed.

## Final validation

- 1717 regression tests passed, no skips or failures.
- 75 pinned-model reference tests passed, no skips or failures.
- 14 upstream dataset/split integration tests passed, no skips or failures.
- Ruff passed for edited Python sources/tests; wheel built successfully.
- The wheel contains the updated directed model and third-party license files.
- The broad HSNN check exercised the existing native Householder fallback on
  Windows; the incompatible optional binary is not treated as verified.

## Server handoff

### Final code review (2026-11-15)

The follow-up review fixed three operational/numerical issues:

- Complex whitening now clamps its theoretically nonnegative covariance
  determinant before taking a square root. Nearly collinear, large float32
  components previously reproduced NaNs. A forward/backward regression covers
  this case. This is a numerical safeguard beyond the upstream implementation.
- Sweep resumption validates ten distinct split rows with finite validation
  and test scores, instead of treating the existence of summary.json as success.
  Corrupt/incomplete results are retried into a new directory; successful retries
  are recognized on subsequent resumes. Summary publication uses atomic rename.
- The verification script fingerprints the current sources, so old successful
  smoke checks cannot silently certify modified code. Failure returns a nonzero
  exit status and report publication uses atomic rename.

Fresh validation: 1724 regression tests and 75 upstream model comparisons passed
with zero failures/skips. Ruff and core type checks passed. XML evidence is in
tmp/dhsnn-review-regression.xml and tmp/dhsnn-review-reference.xml. Previously
reported dataset checks remain applicable; no data files or loaders changed.
An actual CUDA Cora diagonal run also completed all ten frozen splits with
two epochs each and test evaluation enabled. Its summary passes the new
completion validator: tmp/dhsnn-review-cora-ten-splits/summary.json. This is
an execution check, not a converged benchmark or evidence of paper accuracy.

The upload archive now includes complete src/sheaf_mpnn and src/exp trees,
including registries and dataset metadata, to avoid omissions in a hand-selected
runtime patch. It excludes Python caches, virtual environments, and datasets.
Use a new search tag such as review-v3 after this source change; old study
protocol fingerprints intentionally reject mixing implementations.

The local upload archive is tmp/dhsnn-update.tar.gz. Recreate it with:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/package_dhsnn_update.ps1

After stopping old experiments, upload from local PowerShell:

    scp tmp/dhsnn-update.tar.gz root@36.213.113.21:/data1/guo-group/yuan/

On the server, extract from the project root:

    cd /data1/guo-group/yuan/sheaf-mpnn
    tar -xzf ../dhsnn-update.tar.gz
    source .venv-linux/bin/activate
    export CUDA_VISIBLE_DEVICES=2

Use a currently available physical GPU. No dependency changes are needed.
Verify with a fresh output directory, one split and two epochs before launching
complete runs. Old DHSNN checkpoints cannot be resumed; existing result files
remain available. The campaign's mp-v2/pilot-v2 tag creates separate databases
without deleting old studies.
