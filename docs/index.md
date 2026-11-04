# Sheaf-MPNN

**Sheaf-MPNN: A Library for Sheaf Neural Networks as Message Passing** ([arXiv:2611.04670](https://arxiv.org/abs/2611.04670))

`sheaf_mpnn` is a PyTorch / PyTorch Geometric library of **sheaf neural
networks** written in the message-passing paradigm, together with a benchmark
suite: a unified `sheaf` command line, more than 1,300 ready-made presets and
900+ shipped hyperparameter sweeps covering 20+ node-, graph- and link-level
datasets.

## Why this library

Sheaf neural networks attach a vector space (a *stalk*) to every node and a
learned linear *restriction map* to every node-edge incidence, and diffuse
features with the resulting sheaf Laplacian. The original implementations
build dense or block-sparse Laplacians per model and per paper. This library
instead expresses every model as a `torch_geometric.nn.MessagePassing` layer
built on one shared base class, so that

- all families share the same `encoder -> L sheaf convs -> decoder` structure
  and the same restriction-map parameterisations (diagonal, general,
  orthogonal, low-rank, attention-initialised);
- layers take an `edge_index` like any other PyG layer and can be mixed with
  the rest of the PyG ecosystem, including heterogeneous graphs;
- every model is benchmarked under one training protocol, with the same splits,
  early-stopping rule and hyperparameter search.

## Model families

Neural Sheaf Diffusion and its polynomial, connection-Laplacian, deep and
adjacency-normalised variants; Neural Sheaf Propagation; Sheaf Attention
Networks; Bayesian Sheaf Neural Networks; Nonlinear Sheaf Diffusion; Bundle
Neural Networks; directed sheaf diffusion; Cooperative Sheaf Neural Networks;
identity, joint-diffusion and rotation-invariant sheaves; and heterogeneous
mirrors of all of them. See [Models](user-guide/models.md).

## Where to start

- [Installation](installation.md): `pip` from GitHub or a locked `uv` environment.
- [Quick start](quickstart.md): a model in five lines of Python, and a
  cross-validated benchmark run in one command.
- [Tutorials](tutorials.md): ten executed notebooks, from GNN basics to
  directed hypergraph sheaves.
- [API reference](api.md): every public class and function.

## License and citation

Released under the [MIT License](https://github.com/alessioborgi/sheaf-mpnn/blob/main/LICENSE).
If you use the library in your research, please [cite it](citation.md).

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
tutorials
```

```{toctree}
:hidden:
:caption: User guide

user-guide/cli
user-guide/models
user-guide/datasets
user-guide/sweeps
user-guide/benchmark
```

```{toctree}
:hidden:
:caption: Reference

api
changelog
citation
contributing
```
