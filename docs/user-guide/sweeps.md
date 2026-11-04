# Hyperparameter sweeps

Sweeps are Optuna studies described in YAML. One is shipped for every
(dataset, family, variant) under `configs/<dataset>/<family>/<variant>.yaml`,
and each file names the preset it must be run with in its header comment. Run
from the repository root (the dataset cache path is relative):

```bash
sheaf sweep --yaml-path configs/cora/nsd/general.yaml --preset cora_nsd_general
```

:::{warning}
**Always pass the preset documented in the YAML header.** The YAML only
defines the search space and the Optuna settings; the dataset, model variant
and training budget come from the preset. Without it the sweep falls back to
the default configuration (`cora`, `general`) and trains the wrong model.
:::

What a sweep does, in order:

1. Samples `n_trials` configurations (TPE sampler, median pruner). Each trial
   trains `n_seeds_per_trial` folds and reports mean - `std_weight` * std of
   the validation metric. Duplicate configurations are pruned automatically.
2. Prints a progress line per trial and a summary (history, top trials,
   parameter importances) every 10 trials.
3. Retrains the best configuration on all 10 folds and prints test metrics.
4. Writes the best hyperparameters to `<dataset>_<model>_bestconf.yaml` in the
   current directory. Curated results from the authors' sweeps are kept under
   `results/bestconf/`.

Minimal YAML (any field of `ModelConfig`, `RegConfig` or `OptimConfig` can be
swept):

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
  n_seeds_per_trial: 10          # Bodnar protocol: every trial trains all folds
  study_name: nsd-cora
  storage: sqlite:///nsd_cora.db  # optional: resumable / multi-worker study
  # wandb_project: my-project    # optional: one W&B run per trial
  # wandb_entity: my-entity
```

With `storage` set, launching the same command from several machines or GPUs
runs one distributed study; `n_trials` caps the total across all workers. The
launcher scripts under `scripts/` show how the shipped sweeps are batched,
including SLURM submission (`scripts/slurm/sweep.sbatch`).

Shipped configurations set `wandb_project` / `wandb_entity` to the authors' W&B
team. Edit those keys, or run with `WANDB_MODE=disabled`, before launching one.
