# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Add complex directed hypergraph sheaf message passing, four DHSNN map families, a complex-state model, and an audited Cora charge-ablation runner.


### Added

- Undirected Hypergraph Sheaf Neural Networks (`HSNNModel`, `HSNNVariant`) with
  diagonal, general, orthogonal and pure low-rank incidence restriction maps.
  Two-stage message passing applies symmetric sheaf-degree propagation without
  constructing a global Laplacian, with differentiable pseudoinverse degrees,
  optional degree regularization and residuals. Includes independent mathematical
  tests, a synthetic training smoke script and Python API documentation. Existing
  graph implementations and experiment CLI behavior are unchanged.
- `sheaf run --preset hsnn_cora` with audited Cora hypergraph artifacts,
  frozen splits and opt-in final test evaluation. Shares existing typed config
  and registries; the legacy HSNN training script delegates to the packaged
  runner. Existing graph presets retain their behavior.
- Validation-only HSNN Optuna sweeps through `sheaf sweep`, with equal-budget
  diagonal/general YAMLs, a per-GPU launcher, persistent studies and explicit
  export of the selected final-test command.

## [0.1.0]

First packaged release, accompanying
[arXiv:2611.04670](https://arxiv.org/abs/2611.04670). Relicensed under the MIT
License.

### Added

- `sheaf_mpnn`: sheaf neural network layers and models written as
  PyTorch Geometric message-passing layers on a shared `BaseSheafConv`:
  Neural Sheaf Diffusion (diagonal, general, orthogonal, low-rank and
  attention-initialised maps), polynomial, connection-Laplacian, deep and
  adjacency-normalised NSD, Neural Sheaf Propagation, Sheaf Attention
  Networks, Bayesian Sheaf Neural Networks, Nonlinear Sheaf Diffusion, Bundle
  Neural Networks, directed sheaf diffusion, Cooperative Sheaf Neural
  Networks, identity, joint-diffusion and rotation-invariant sheaves, and the
  gauge-invariant `SheafPool` readout.
- `hetsheaf_mpnn`: heterogeneous mirrors of every family, with restriction
  maps conditioned on node and edge types.
- `exp`: the `sheaf` command line with 10-fold cross-validated training,
  1,300+ presets, YAML-driven Optuna sweeps (900+ shipped configurations),
  loaders for 20+ node-, graph- and link-level datasets, optional Weights &
  Biases logging, and an efficiency benchmark against the original NSD code.
- Ten tutorial notebooks, committed with their outputs.
- Documentation site with a user guide and an API reference generated from the
  docstrings.

[Unreleased]: https://github.com/alessioborgi/sheaf-mpnn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alessioborgi/sheaf-mpnn/releases/tag/v0.1.0
