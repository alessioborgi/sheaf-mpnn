# Quickstart

## Use the layers directly

Each NSD convolution layer expects node features as a flat vector of $d \cdot \text{channels}$ values per node, input shape `[N, d * in_channels]`. The stalk dimension $d$ controls the per-node vector space assigned by the sheaf; larger $d$ allows richer inter-node maps at the cost of more parameters. Output shape is `[N, d * out_channels]`.

```python
import torch
from sheaf_mpnn.nsd import DiagonalNSDConv

x = torch.randn(10, 4 * 16)          # N=10 nodes, stalk dim d=4, 16 channels
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

conv = DiagonalNSDConv(in_channels=16, out_channels=16, d=4)
h = conv(x, edge_index)               # shape: [10, 4 * 16]
```

The `variant` argument selects the restriction-map family: `"diagonal"` ($d$ params/edge endpoint), `"general"` ($d^2$ params), `"orthogonal"` ($\tfrac{d(d-1)}{2}$ params, norm-preserving).

## Run a preset

10-fold cross-validation on Cora with best-known hyperparameters:

```bash
sheaf run --preset cora
```

Override individual fields on top of a preset:

```bash
sheaf run --preset cora --model.num-layers 4 --optim.lr 5e-3
```

Fully manual configuration:

```bash
sheaf run \
    --dataset.name cora \
    --model.variant general \
    --model.d 4 \
    --model.num-layers 2
```

All flags: `sheaf run --help`. The legacy `python -m exp.run` invocation still works.

## Download splits

Fetch the canonical Geom-GCN train/val/test splits (done automatically on first run):

```bash
sheaf splits                          # all datasets
sheaf splits --datasets cora texas    # specific datasets only
sheaf splits --source generate        # generate locally instead
```

## Hyperparameter sweep

Sweeps are YAML-driven. Create a config file and run:

```bash
sheaf sweep --yaml-path sweep.yaml --preset cora
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
  # storage: sqlite:///sweep.db  # uncomment for distributed / resumable sweeps
```
