# Quick start

## A model from Python

Every model takes node features and an `edge_index`, like any PyG model:

```python
import torch
from sheaf_mpnn.nsd import NSDModel, NSDVariant

x = torch.randn(100, 32)                       # [N, in_channels]
edge_index = torch.randint(0, 100, (2, 400))   # [2, E]

model = NSDModel(
    in_channels=32, out_channels=4,
    stalk_dim=3, hidden_dim=16, num_layers=2,
    variant=NSDVariant.GENERAL, alpha=1.0,
)
logits = model(x, edge_index)                  # [N, 4]
```

## A single sheaf convolution

Convolutions take the node state twice: flat (`x_feat`, used to generate the
restriction maps) and as stalks (`x_stalk`, the signal being diffused):

```python
from torch import nn
from sheaf_mpnn.nsd import GeneralNSDConv

d, hidden_dim = 4, 16
encoder = nn.Linear(32, d * hidden_dim)
layer = GeneralNSDConv(
    stalk_dim=d, in_channels=hidden_dim, hidden_dim=hidden_dim,
    context_dim=d * hidden_dim, alpha=1.0,
)

x_stalk = encoder(x).view(-1, d, hidden_dim)    # [N, d, hidden_dim]
x_feat = x_stalk.reshape(x_stalk.size(0), -1)   # [N, d * hidden_dim]
x_stalk = layer(x_feat, x_stalk, edge_index)    # [N, d, hidden_dim]
```

`sheaf_mpnn.__all__` lists every public model, convolution and variant enum;
the [API reference](api.md) documents them all.

## A benchmark run from the command line

With the `experiments` extra installed, train Neural Sheaf Diffusion on Texas
with its best-known configuration (10-fold cross-validation, early stopping on
validation loss):

```bash
sheaf run --preset texas
```

A sanity check that finishes in seconds on CPU: cap the epochs and folds. Every
preset field can be overridden this way.

```bash
sheaf run --preset texas --optim.epochs 3 --cv.folds 2
```

The run prints the dataset statistics, the split sizes, a parameter summary,
one row per fold and the mean and standard deviation of the test metric.

## Loading a benchmark dataset

```python
from exp.data import load_dataset

data, info = load_dataset("texas", root="exp/data")   # PyG Data + metadata
print(data.num_nodes, info.num_classes, info.metric)
```

Next: the [command line guide](user-guide/cli.md) and the
[tutorial notebooks](tutorials.md).
