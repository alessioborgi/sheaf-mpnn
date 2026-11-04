# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Generate the BSNN sweep YAMLs under ``configs/<dataset>/bsnn/``.

One sweep per (dataset, variant) pair, mirroring the NSD sweep layout: the
search space is shared, only the identity fields (header, study name, storage)
change. Run from the repository root:

    uv run python scripts/gen_bsnn_sweeps.py
"""

from __future__ import annotations

from pathlib import Path

VARIANTS = ("diagonal", "general", "orthogonal", "low_rank")

# Orthogonal-map parameterization only matters for the SO(d) variant; cayley
# is exactly SO(d), householder matches the reference BSNN configuration.
ORTH_BLOCK = """
  orth_strategy:
    type: categorical
    choices: [cayley, fasthpp, householder]
"""

# Rank bound of the factored maps F = A B^T; only the low_rank variant reads
# it. Same grid as the NSD low_rank sweeps.
RANK_BLOCK = """
  rank:
    type: categorical
    choices: [1, 2, 4]
"""

# Per-variant extra search-space knobs appended after the shared grid.
EXTRA_BLOCKS = {"orthogonal": ORTH_BLOCK, "low_rank": RANK_BLOCK}

TEMPLATE = """\
# BSNN sweep on {dataset} - {variant} restriction map.
#
# Bayesian Sheaf Neural Network (Gillespie et al., 2025, arXiv:2410.09590):
# the sheaf is a latent random variable, training maximizes the ELBO with a
# cyclically annealed KL term, and evaluation ensembles stochastic passes.
#
# Run with:
#   python -m exp.sweeps.sweep --yaml-path configs/{dataset}/bsnn/{variant}.yaml \\
#     --preset {dataset}_bsnn_{variant}

model: bsnn

search_space:
  stalk_dim:
    type: categorical
    choices: [2, 3, 4, 5, 6]
  hidden_dim:
    type: categorical
    choices: [8, 16, 32, 64]
  num_layers:
    type: int
    low: 2
    high: 6
  alpha:
    type: float
    low: 0.01
    high: 4.0
    log: true
  input_dropout:
    type: categorical
    choices: [0.0, 0.2, 0.5, 0.7]
  dropout:
    type: categorical
    choices: [0.0, 0.2, 0.5, 0.7]
  lr:
    type: categorical
    choices: [0.02, 0.01, 0.002]
  weight_decay:
    type: float
    low: 0.0000001
    high: 0.01
    log: true
  sheaf_decay:
    type: float
    low: 0.00000001
    high: 0.01
    log: true

  add_lp:
    type: categorical
    choices: [false, true]
  add_hp:
    type: categorical
    choices: [false, true]
  sparse_learner:
    type: categorical
    choices: [false, true]
  second_linear:
    type: categorical
    choices: [false, true]
  learn_alpha:
    type: categorical
    choices: [false, true]

  # ELBO knobs: maximum KL weight, annealing schedule, cycle length, and the
  # number of stochastic forward passes ensembled at evaluation time.
  kl_weight:
    type: float
    low: 0.01
    high: 10.0
    log: true
  kl_anneal:
    type: categorical
    choices: [cyclical, constant]
  kl_cycle:
    type: categorical
    choices: [20, 40, 80]
  ensemble_size:
    type: categorical
    choices: [1, 3, 5]
{extra_block}
config:
  n_trials: 250
  n_seeds_per_trial: 10  # Bodnar protocol: every trial trains all 10 folds
  std_weight: 0.0            # penalty weight on std; 0 = optimise pure mean
  study_name: bsnn-{dataset}-{variant}
  seed: 42
  storage: sqlite:///bsnn_{dataset}_{variant}.db   # for distributed sweeps
  wandb_project: sheaf-mpnn-runs       # uncomment to log to Weights & Biases
  wandb_entity: adavit
"""


def main() -> None:
    """Write one BSNN sweep YAML per (dataset, variant) pair."""
    configs = Path(__file__).resolve().parent.parent / "configs"
    # BSNN targets the undirected node-classification benchmarks: every
    # dataset that already carries an NSD sweep gets the BSNN grid.
    datasets = sorted(p.parent.name for p in configs.glob("*/nsd"))
    for dataset in datasets:
        out_dir = configs / dataset / "bsnn"
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            (out_dir / f"{variant}.yaml").write_text(
                TEMPLATE.format(
                    dataset=dataset,
                    variant=variant,
                    extra_block=EXTRA_BLOCKS.get(variant, ""),
                )
            )
            print(f"wrote configs/{dataset}/bsnn/{variant}.yaml")


if __name__ == "__main__":
    main()
