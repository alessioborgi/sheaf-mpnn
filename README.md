# Sheaf Neural Networks – PyTorch Implementation

A clean PyTorch / PyG implementation library of  **Sheaf Neural Networks** comprising all variants and a benchmark suite with 20+ Datasets.

**Copyright © 2026, _Sheaf Neural Networks as Message Passing_.**
Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite, Mario Severino,
Emanuele Mule, Dario Loi, Francesco Restuccia, Fabrizio Silvestri, and Pietro Liò.


![Sheaf Neural Networks as Message Passing](img/MPSNN-1.png)

## Quick Start

```bash
git clone https://github.com/<you>/sheaf-mpnn.git
cd pytorch-SheafNeuralNetworks
uv sync && source .venv/bin/activate

# 10-fold CV on Cora with the best-known config (data downloaded automatically)
sheaf run --preset cora
```

Use the library directly:


```python
from sheaf_mpnn.nsd import NSDModel, NSDVariant

model = NSDModel(
    in_channels=1433, out_channels=7,
    stalk_dim=4, hidden_dim=16, num_layers=2,
    variant=NSDVariant.GENERAL, alpha=1.0,
)
logits = model(x, edge_index)  #  [N, 7]
```

Use the layers directly when you want to build your own architecture:

```python
from torch import nn

from sheaf_mpnn.nsd import GeneralNSDConv

d, hidden_dim = 4, 16
encoder = nn.Linear(1433, d * hidden_dim)
layer = GeneralNSDConv(
    stalk_dim=d,
    in_channels=hidden_dim,
    hidden_dim=hidden_dim,
    context_dim=d * hidden_dim,
    alpha=1.0,
)

x_stalk = encoder(x).view(-1, d, hidden_dim)       # [N, d, hidden_dim]
x_feat = x_stalk.reshape(x_stalk.size(0), -1)      # [N, d * hidden_dim]
x_stalk = layer(x_feat, x_stalk, edge_index)       # [N, d, hidden_dim]
```

## Installation

```bash
uv sync                   # core dependencies
uv sync --extra wandb     # + Weights & Biases / Optuna-WandB
```

**Requirements:** Python ≥ 3.13, PyTorch ≥ 2.4, PyTorch Geometric ≥ 2.5, Lightning ≥ 2.3.

## Datasets

All 14 datasets download automatically into `exp/data/` on first use.

| Dataset | Nodes | Edges | Features | Classes | Metric | Split |
|---------|------:|------:|--------:|--------:|:------:|-------|
| `cora` | 2 708 | 10 556 | 1 433 | 7 | Acc | Geom-GCN |
| `citeseer` | 3 327 | 9 104 | 3 703 | 6 | Acc | Geom-GCN |
| `chameleon` | 2 277 | 36 101 | 2 325 | 5 | Acc | Geom-GCN |
| `chameleon_filtered` | 890 | 8 854 | 2 325 | 5 | Acc | Geom-GCN filtered |
| `squirrel` | 5 201 | 217 073 | 2 089 | 5 | Acc | Geom-GCN |
| `squirrel_filtered` | 2 223 | 47 138 | 2 089 | 5 | Acc | Geom-GCN filtered |
| `cornell` | 183 | 298 | 1 703 | 5 | Acc | Geom-GCN |
| `texas` | 183 | 325 | 1 703 | 5 | Acc | Geom-GCN |
| `film` | 7 600 | 30 019 | 932 | 5 | Acc | Geom-GCN |
| `amazon_ratings` | 24 492 | 186 100 | 300 | 5 | Acc | Platonov |
| `minesweeper` | 10 000 | 39 402 | 7 | 2 | ROC-AUC | Platonov |
| `questions` | 48 921 | 153 540 | 301 | 2 | ROC-AUC | Platonov |
| `roman_empire` | 22 662 | 32 927 | 300 | 18 | Acc | Platonov |
| `tolokers` | 11 758 | 519 000 | 10 | 2 | ROC-AUC | Platonov |

## Model Variants

All variants share the same `encoder → NSD layers → decoder` architecture; only the restriction-map parameterisation differs.

| Variant | Flag | Params / edge | Notes |
|---------|------|:-------------:|-------|
| Diagonal | `--model.variant diagonal` | O(d) | Lightweight baseline |
| General | `--model.variant general` | O(d²) | Most expressive |
| Orthogonal | `--model.variant orthogonal` | O(d(d−1)/2) | Numerically stable via Cayley transform |

## Running Experiments

### Presets

Every dataset has a built-in preset. Any field can still be overridden:

```bash
sheaf run --preset cora
sheaf run --preset texas --model.variant orthogonal --model.stalk-dim 5
```

Run `sheaf run --help` for the full list of flags. The legacy `python -m exp.run` invocation still works.

### Splits

Download the canonical Geom-GCN splits (done automatically on first run, but can be pre-fetched):

```bash
sheaf splits                          # all NPZ-split datasets
sheaf splits --datasets cora texas    # specific datasets only
sheaf splits --source generate        # generate locally instead
```

### Weights & Biases

```bash
sheaf run --preset cora \
    --wandb.enabled --wandb.entity your_entity --wandb.project nsd-cora
```

### Hyperparameter Sweeps

Sweeps are YAML-driven. Create a config file specifying the model, search space, and
Optuna settings, then run:

```bash
sheaf sweep --yaml-path sweep.yaml --preset cora

# Distributed sweep  add storage to the YAML under config:
#   config:
#     storage: sqlite:///sweep.db
```

Example `sweep.yaml`:

```yaml
model: nsd
search_space:
  variant:
    type: categorical
    choices: [diagonal, general, orthogonal]
  stalk_dim:
    type: int
    low: 2
    high: 8
  lr:
    type: float
    low: 0.0001
    high: 0.1
    log: true
config:
  n_trials: 100
  study_name: nsd-cora
```

Override the preset's dataset directly in the YAML with a `dataset:` block, or run
distributed sweeps by adding `storage: sqlite:///sweep.db` under `config:`.
## Development & Quality Control

We use `uv` for dependency management and `prek` (a Rust-based git-hook framework) to ensure code quality.

To run the full suite of checks (linting, formatting, type checking, and unit tests) in one command:

```bash
uv run prek run --all-files
uv run pytest # Run the unit test suite
uv run pytest -m integration # Run the integration tests
```

Alternatively, you can run the individual tools:

```bash
uv run ty check                  # Perform static type checking (via mypy)
uv run ruff check --fix --unsafe-fixes # Lint and apply all automatic fixes
uv run ruff format .              # Standardize code formatting
uv run pytest                     # Run the unit test suite
uv run pytest -m integration      # Run the integration tests
```

## Citation

If you use this library in your research, please cite our forthcoming paper (to be soon published):

```bibtex
@unpublished{borgi2026sheafneuralnetworks,
  title  = {Sheaf Neural Networks as Message Passing},
  author = {Borgi, Alessio and Onorato, Gabriele and Braithwaite, Luke
            and Severino, Mario and Mule, Emanuele and Loi, Dario
            and Restuccia, Francesco and Silvestri, Fabrizio and Liò, Pietro},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```
