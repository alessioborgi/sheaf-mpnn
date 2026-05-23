# Sheaf Neural Networks – PyTorch Implementation

A clean PyTorch / PyG implementation library of  **Sheaf Neural Networks** comprising all variants and a benchmark suite with 20+ Datasets.

**Copyright © 2026, _Sheaf Neural Networks as Message Passing_.**
**Authors:** _Alessio Borgi_, _Gabriele Onorato_, _Luke Braithwaite_, _Mario Severino_,
_Emanuele Mule_, _Dario Loi_, _Francesco Restuccia_, _Fabrizio Silvestri_, and _Pietro Liò_.


![Sheaf Neural Networks as Message Passing](img/MPSNN-1.png)

## Quick Start

```bash
git clone https://github.com/<you>/pytorch-SheafNeuralNetworks.git
cd pytorch-SheafNeuralNetworks
uv sync && source .venv/bin/activate

# 10-fold CV on Cora with the best-known config (data downloaded automatically)
python -m exp.run --preset cora
```

Use the library directly:

```python
from sheaf_mpnn.nsd import NSDModel, NSDVariant

model = NSDModel(
    in_channels=1433, out_channels=7,
    stalk_dim=4, hidden_dim=16, num_layers=2,
    variant=NSDVariant.GENERAL, alpha=1.0,
)
logits = model(x, edge_index)  # → [N, 7]
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
python -m exp.run --preset cora
python -m exp.run --preset texas --model.variant orthogonal --model.stalk-dim 5
```

Run `python -m exp.run --help` for the full list of flags.

### Weights & Biases

```bash
python -m exp.run --preset cora \
    --wandb.enabled --wandb.entity your_entity --wandb.project nsd-cora
```

### Hyperparameter Sweeps

Sweeps are YAML-driven. Create a config file specifying the model, search space, and
Optuna settings, then run:

```bash
python -m exp.sweeps.sweep --yaml-path sweep.yaml --preset cora
```

The sweep CLI only accepts `--yaml-path` and `--preset`. All other options (number of
trials, epochs, folds, storage) are set inside the YAML file.

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
  # storage: sqlite:///sweep.db  # uncomment for distributed / resumable sweeps
```

Override the preset's dataset directly in the YAML with a `dataset:` block.

## Running Tests

```bash
uv run pytest              # full suite
uvx ruff check .           # lint
uvx ruff format --check .  # formatting
uvx prek install           # install git hooks (defined in prek.toml)
```

## Citation

If you use this library in your research, please cite our forthcoming paper:

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
