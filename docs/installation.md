# Installation

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11, 3.12 or 3.13 | Tested in CI on all three. |
| PyTorch >= 2.6 and PyTorch Geometric >= 2.6 | Installed automatically; CPU and CUDA builds both work. |
| Linux, macOS or Windows | Tested in CI on all three. |

## With pip

The core library (`sheaf_mpnn` and `hetsheaf_mpnn`) only depends on PyTorch,
PyTorch Geometric, NumPy and SciPy:

```bash
pip install "sheaf_mpnn @ git+https://github.com/alessioborgi/sheaf-mpnn"
```

Optional extras add the rest of the tooling:

| Extra | Adds |
|---|---|
| `experiments` | The `sheaf` command line: Lightning training loop, dataset loaders, presets, Optuna sweeps |
| `wandb` | Weights & Biases logging for runs and sweeps |
| `benchmark` | The efficiency benchmark against the original NSD code (submitit, matplotlib) |
| `notebooks` | Jupyter and matplotlib for the tutorial notebooks |
| `householder` | The compiled `torch-householder` kernel for orthogonal maps (a numerically identical native fallback is used without it) |
| `all` | Everything above |

```bash
pip install "sheaf_mpnn[experiments] @ git+https://github.com/alessioborgi/sheaf-mpnn"   # library + sheaf CLI
pip install "sheaf_mpnn[all] @ git+https://github.com/alessioborgi/sheaf-mpnn"           # everything
```

To get a CUDA build of PyTorch, install `torch` first following
<https://pytorch.org/get-started/locally/>, then install `sheaf_mpnn`.

The optional compiled PyG extensions (`torch-scatter`, `torch-sparse`,
`pyg-lib`) are not required. PyG uses them for speed when they are present;
see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## From source with uv (reproducible environment)

The repository ships a `uv.lock` that pins a fully reproducible environment
(Python 3.13, PyTorch 2.8.0, CUDA 12.8 on Linux/Windows, CPU on macOS, plus the
compiled PyG extensions):

```bash
git clone https://github.com/alessioborgi/sheaf-mpnn.git
cd sheaf-mpnn
uv sync                       # creates .venv with every extra and the dev tools
source .venv/bin/activate     # or prefix every command with `uv run`
sheaf --help
```

The efficiency benchmark additionally needs the git submodule under
`external/`; see [Efficiency benchmark](user-guide/benchmark.md).

## Datasets

Datasets are downloaded on first use and cached under `exp/data/` (relative to
the working directory). No account or token is needed. The directed DSNN
variants of the citation and Platonov graphs, whose edge lists, labels and
masks differ from the public originals, are served from the repository's
`dataset-assets-v1` GitHub release. Set `SHEAF_DATASET_REPO=<owner>/<name>` to
download them from a fork or mirror instead.
