# Directed hypergraph sheaf message passing

DHSNNModel now follows the full DSHN architecture at reference commit
d94f81bc7d86bd8d41e416083063f3446c1ad796, using incidence-wise message passing
instead of assembling the global sparse propagation matrix. There is one
production model, not separate native/reference implementations.

## Architecture

Input is lifted to x + i*x.detach() and both components use a shared bias-free
projection. Restriction maps use stalk-mean contexts, a bias-free predictor and
sigmoid by default. MLP_var1 uses the initial projected mean-hyperedge features;
MLP_var2 recomputes their mean from current states; MLP_var3 applies a trainable
projection before sum aggregation. Static maps are predicted once PER FORWARD;
dynamic_sheaf=True uses independent per-layer predictors. Nothing differentiable
is cached across optimizer steps.

Each layer optionally mixes stalk coordinates, mixes channels, applies complex
batch whitening, propagates, then adds residuals and complex bias. ReLU and
shared-mask dropout occur only between layers. ReLU retains nonnegative real
parts, matching the reference. A configurable real MLP sees both final complex
components. Hyperedge tasks sum node representations BEFORE classification.

Default architecture settings: static MLP_var1, sigmoid, symmetric block
normalization, identity augmentation, complex whitening, no left projection,
two-layer classifier of width 64 with LayerNorm after ReLU. Library widths,
layer counts and training budgets are NOT paper-selected hyperparameters.

## Transport and gradients

The library's edge-by-node B uses negative tail phase. Official node-by-edge H
uses positive tail phase and equals B-adjoint when its real blocks are F.T.
Opposite phase signs alone are not an error. D_left H E-inverse H-adjoint
D_right X is factored into node-to-edge and edge-to-node aggregation.

Symmetric modes apply both node factors; nonsymmetric modes only the left.
Full maps use expanded-incidence edge degree d*|e|; diagonal maps use |e|.
General block modes always augment with identity and detach the inverse
degree power, matching the actual upstream implementation. Orthogonal block
modes become degree modes and retain unconditional identity augmentation.

The standalone convolution's norm_type=None exposes the mathematical
normalized fixed-map operator. Complete models always select explicit reference
normalization; this is not a second legacy architecture.

Low-rank, FastH/FastH++ and componentwise ELU are library extensions.
Householder uses the native compact product, avoiding a compiled extension.
Random hyperedge initialization and DSHN-Light are not exposed in this release.

## CLI

    sheaf run --preset cora-hypergraph --model.type dhsnn --hardware.cuda 0
    sheaf run --preset molecular-2-hypergraph --model.type dhsnn --model.variant general

Ten frozen splits and test evaluation are defaults. Validation selects
checkpoints. Use --hypergraph.no-evaluate-test for architecture debugging.
Useful model options:

    --model.dhsnn-dynamic-sheaf
    --model.dhsnn-map-activation sigmoid
    --model.dhsnn-prediction MLP_var1
    --model.dhsnn-norm-type sym_block_norm
    --model.dhsnn-left-projection
    --model.no-dhsnn-input-norm
    --model.dhsnn-classifier-hidden 64
    --model.dhsnn-classifier-layers 2
    --model.dhsnn-classifier-norm ln

Search records test scores but selects only by validation, then automatically
evaluates the best configuration on ten splits. All directed fields reach the
registry, runner, sweep builder and final-test export. A limited campaign:

    python scripts/run_hypergraph_campaign.py --dataset cora-hypergraph --model dhsnn --variant diagonal --trials 20 --splits 3 --tag pilot-v2

Use a new tag for a changed protocol; reuse the tag to resume. Default mp-v2
tags isolate new studies from old databases. Shipped search spaces are library
spaces, not a verified copy of all official W&B experiments.

## Verification and compatibility

    python scripts/prepare_dshn_reference.py
    python -m pytest tests/integration/test_dhsnn_reference.py -m integration --no-cov

Reference downloads are explicit and cached under ignored tmp/. Tests never
download code; missing caches are reported as skips. See
[detailed audit](dhsnn-official-audit.md) for numerical errors and measured scope.
Old DHSNN checkpoints are incompatible; use fresh results/studies. Dataset
artifacts and ordinary HSNN remain unchanged. Reference attribution and
Apache-2.0 terms are in third_party/dshn/.
