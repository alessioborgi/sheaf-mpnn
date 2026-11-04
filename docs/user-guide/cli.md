# The `sheaf` command line

Installed with the `experiments` extra.

```
sheaf run        10-fold cross-validation on a node-classification dataset
sheaf splits     download or generate the Geom-GCN train/val/test splits
sheaf sweep      YAML-driven Optuna hyperparameter sweep
sheaf benchmark  submit / collect / report the efficiency benchmark
```

Add `--help` after any sub-command for its flags. `python -m exp.run` is
equivalent to `sheaf run`.

## Presets

A preset is a complete, named configuration (dataset, model family,
restriction-map variant, training budget). There are over 1,300 of them, all
generated from two tables in `src/exp/registries/presets.py`. Keys follow one
pattern:

| Key | Meaning | Example |
|---|---|---|
| `<dataset>` | NSD with that dataset's default variant | `cora`, `texas`, `roman_empire` |
| `<dataset>_<family>_<variant>` | One family/variant on that dataset | `cora_nsd_orthogonal`, `texas_nsp_diagonal`, `questions_bsnn_general` |
| `<dataset>_<family>` | Families with no variant grid | `cora_conn_nsd`, `cora_id_sheaf`, `cora_bunn`, `cora_csnn`, `cora_sheafan` |
| `<dataset>_directed_nsd_<variant>` | Directed NSD; also sets `--directed` | `cora_directed_nsd_general` |

The five Pei et al. datasets (`texas`, `wisconsin`, `cornell`, `chameleon`,
`squirrel`) use the exact Bodnar et al. hyperparameters for their bare preset.

List the presets available for a dataset:

```bash
python -c "from exp.registries import preset_registry as r; print(sorted(k for k in r.list_keys() if k.startswith('cora')))"
```

Passing an unknown preset also prints the full list of known keys.

## Overriding fields

Flags mirror the nested dataclasses in `exp.config`: `--<group>.<field>`.
Choices are lower-case; `--model.type` and `--model.pooling` also accept
upper-case enum names.

```bash
sheaf run --preset texas --model.variant orthogonal --model.stalk-dim 5
sheaf run --preset cora --optim.lr 0.005 --optim.epochs 500 --cv.folds 5 --hardware.cuda 1
sheaf run --preset cora_directed_nsd_general          # trains on the directed graph
sheaf run --preset cora --directed                    # keep edges directed for any model
```

| Group | Flags (abridged) |
|---|---|
| `--dataset.*` | `name`, `root` (default `exp/data`); top-level `--directed` |
| `--model.*` | `type`, `variant`, `stalk-dim`, `hidden-dim`, `num-layers`, `alpha`, `learn-alpha`, `rank`, `orth-strategy`, plus per-family knobs (`bunn-*`, `nlsd-*`, `kl-*`, `gnn-*`, ...) |
| `--reg.*` | `input-dropout`, `dropout` |
| `--optim.*` | `lr`, `weight-decay`, `sheaf-decay`, `epochs`, `early-stopping`, `stop-strategy {loss,acc}` |
| `--cv.*` | `folds`, `seed`, `min-acc` |
| `--hardware.*` | `cuda`, `num-workers`, `pin-memory` |
| `--wandb.*` | `enabled`, `entity`, `project` |

The full set of fields is documented in the API reference for
{py:mod}`exp.config`.

## Heterogeneous and graph-level tasks

HGB heterogeneous datasets and TU graph-classification datasets have their own
runners. They accept the same `--preset` and override flags:

```bash
# HGB node classification (imdb, dblp, acm, freebase); het_* model types
python -m exp.het_run --preset dblp --model.type het_nsd
python -m exp.het_run --preset dblp_het_nlsd_general

# TU graph classification; pooling via --model.pooling
python -m exp.gc_run --preset mutag
```

## Splits

The canonical Geom-GCN 10-fold splits for the nine NPZ-split datasets are
fetched automatically on first run. To pre-fetch or regenerate them:

```bash
sheaf splits                          # all nine NPZ-split datasets
sheaf splits --datasets cora texas    # a subset
sheaf splits --source generate        # stratified 48/32/20 splits made locally
```

## Weights & Biases

Logging is optional and off by default (install the `wandb` extra). Add
`--wandb.enabled` to any `sheaf run`, `exp.het_run` or `exp.gc_run` command:

```bash
sheaf run --preset cora --wandb.enabled --wandb.entity <entity> --wandb.project <project>
```

Each cross-validation fold becomes its own W&B run, grouped by configuration.
`WANDB_MODE=offline` logs locally and `WANDB_MODE=disabled` turns logging off
regardless of the configuration.
