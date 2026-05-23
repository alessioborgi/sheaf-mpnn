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
python -m exp.run --preset cora
```

Override individual fields on top of a preset:

```bash
python -m exp.run --preset cora --model.num-layers 4 --optim.lr 5e-3
```

Fully manual configuration:

```bash
python -m exp.run \
    --dataset.name cora \
    --model.variant general \
    --model.d 4 \
    --model.num-layers 2
```

All flags: `python -m exp.run --help`.

## Hyperparameter sweep

```bash
python -m exp.sweep --preset cora --n-trials 100
```

Distributed sweeps share an SQLite study:

```bash
python -m exp.sweep --preset cora \
    --storage sqlite:///studies/cora.db \
    --study-name cora-v1
```
