# Cooperative Sheaf Neural Networks (CSNN)

This package implements **Cooperative Sheaf Neural Networks**, a directed sheaf
model that learns **conformal** restriction maps.

## Variants

| Variant | Flag | Restriction map | Notes |
|---------|------|-----------------|-------|
| Conformal | `--model.variant conformal` | $c\,Q$, $c \in (0, 10]$ | Default; the reference CSNN |
| Non-conformal | `--model.variant non_conformal` | $Q$ | Scale pinned to 1 |

## Quick Start

```python
from sheaf_mpnn.csnn import CSNNModel, CSNNVariant

model = CSNNModel(
    in_channels=300, out_channels=18,
    stalk_dim=4, hidden_dim=32, num_layers=5,
    variant=CSNNVariant.CONFORMAL,
    gnn_layers=3, gnn_hidden=32, gnn_default=2,
    orth_strategy="householder",
)
logits = model(x, edge_index)  # [N, 18]
```

Layer-level use follows the package convention, with `x_feat` and `x_stalk` being
the same tensor in two shapes (the caller applies dropout once so the sheaf
learners and the skip term share a mask):

```python
from sheaf_mpnn.csnn import CSNNConv

d, hidden_dim = 4, 32
layer = CSNNConv(stalk_dim=d, hidden_dim=hidden_dim, gnn_layers=1)

x_feat = x_stalk.reshape(x_stalk.size(0), -1)   # [N, d * hidden_dim]
out = layer(x_feat, x_stalk, edge_index)        # [N, d * hidden_dim]
```

### Sheaf-learner backbones

`gnn_default` selects the map generator:

| `gnn_default` | Backbone |
|---|---|
| `1` | `GraphSAGE` with layer norm |
| `2` | `GraphSAGE` with GELU, projection and dropout 0.2 |
| `0` | custom stack from `gnn_type` (the only path that consumes `edge_attr`) |

Setting `gnn_layers=0` replaces the GNN with a two-layer MLP; `gnn_hidden` still
sizes it.

Two custom-path constraints are enforced with clear errors: `GPS` requires
`linear_emb=True` (`GPSConv` preserves its input width, so only the
embed/project wrapper reaches the map-parameter width), and `NNConv` requires
`edge_attr` -- which the node-classification pipeline never provides.

`forward` accepts optional `edge_attr` and `pe` arguments. Neither is used by the
reproduction datasets, but both are threaded through for future ones. `pe_size`
must match the width of the `pe` tensor.

## Package Structure

| File | Contents |
|---|---|
| `csnn_layers.py` | `CSNNConv`, `ConformalSheafLearner` |
| `csnn_model.py` | `CSNNModel`, `CSNNVariant` |
| `gnn_builder.py` | `get_layer_type`, `build_gnn`, `SumGNN` |
| `../../tests/csnn/` | Unit tests |

## Reproduction

Presets `<dataset>_csnn` transcribe the reference hyperparameters for
`roman_empire`, `amazon_ratings`, `minesweeper`, `questions`, `squirrel` and
`tolokers`.

```bash
sheaf run --preset roman_empire_csnn
bash scripts/csnn/smoke_csnn.sh      # 1 fold x 5 epochs, all datasets
bash scripts/csnn/run_csnn_repro.sh  # full 10-fold reproduction
```

`run_csnn_repro.sh` documents the reference numbers and the known data-pipeline
deviations (graph directedness, the `squirrel` dataset identity, and the
1-logit-vs-2-logit binary head).

Hyperparameter sweeps follow the same grid as every other family: one
`configs/<dataset>/csnn/conformal.yaml` per node-classification dataset (all
16), runnable individually
(`sheaf sweep --yaml-path ... --preset <dataset>_csnn`) or in sequence via
`bash scripts/sweep_platonov_family.sh csnn`. Datasets outside the six the
paper tunes anchor their sweep on a documented `<dataset>_csnn` preset that
mirrors the dataset's shared hyperparameter row.

## Mathematical Properties & Testing

- **Edgeless identity** -- at initialisation a graph with no edges is mapped to
  itself exactly, pinning the residual and the message convention.
- **Orthogonality** -- restriction maps satisfy $QQ^\top = I$ within 1e-5.
- **Conformal bound** -- the scale lies in $(0, 10]$ via `exp(clamp_max(·, log 10))`.
- **Structural parity** -- parameter counts match the reference implementation
  exactly on all six datasets.
- **Decoder contract** -- with `decoder` swapped for `nn.Identity`, `forward`
  returns the flat stalk representation, as graph classification requires.

```bash
uv run pytest tests/csnn/
```

## Notes on the port

The reference `ConformalSheafLearner` (the `ICLR-Reproducibility` release, the
code the paper's experiments ran) accepts a `sheaf_act` argument and then
always applies `tanh`; that dead parameter is not carried over. The reference
also sizes some learner branches as `(hidden_dim + pe_size) * stalk_dim`, which
only agrees with the concatenated width when `pe_size == 0`; this implementation
uses the consistent `stalk_dim * hidden_dim + pe_size` so `pe_size > 0` works.

The dense (explicit sparse-Laplacian) CSNN variant is deliberately not ported.
