# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Generate the NLSD sweep YAMLs under ``configs/<dataset>/nlsd/``.

One sweep per (dataset, variant) pair, mirroring the NSD sweep layout: the
search space is shared, only the identity fields (header, study name, storage)
change. Run from the repository root:

    uv run python scripts/gen_nlsd_sweeps.py
"""

from __future__ import annotations

from pathlib import Path

VARIANTS = ("diagonal", "orthogonal", "general", "low_rank")

# Orthogonal-map parameterization only matters for the SO(d) variant; the
# reference NLSD uses the orthogonal family, with edge weights sweepable.
ORTH_BLOCK = """
  orth_strategy:
    type: categorical
    choices: [cayley, fasthpp, householder]
  edge_weights:
    type: categorical
    choices: [false, true]
"""

RANK_BLOCK = """
  rank:
    type: categorical
    choices: [1, 2, 4]
"""

TEMPLATE = """\
# NLSD sweep on {dataset} - {variant} restriction map.
#
# Nonlinear Sheaf Diffusion (Zaghen et al., GRaM @ ICML 2024): a learned
# nonlinearity Phi sits inside the sheaf Laplacian, L^Phi = delta^T Phi delta.
# "mlp" is the paper's learned Phi; the threshold variants keep the
# hand-crafted opinion-dynamics gate but learn its threshold.
#
# Run with:
#   python -m exp.sweeps.sweep --yaml-path configs/{dataset}/nlsd/{variant}.yaml \\
#     --preset {dataset}_nlsd_{variant}

model: nlsd

search_space:
  stalk_dim:
    type: categorical
    choices: [2, 3, 4, 5]
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

  sparse_learner:
    type: categorical
    choices: [false, true]
  second_linear:
    type: categorical
    choices: [false, true]
  learn_alpha:
    type: categorical
    choices: [false, true]

  # Which Phi sits between the coboundary and its transpose. "identity" is
  # linear NSD and is swept by the nsd family already, so it is left out.
  nlsd_nonlinearity:
    type: categorical
    choices: [mlp, scalar_threshold, edge_threshold]
  # Potential whose derivative gates the threshold variants; ignored by "mlp".
  nlsd_psi:
    type: categorical
    choices: [linear, cos]
  # Depth of the learned Phi; only read by "mlp".
  nlsd_mlp_layers:
    type: categorical
    choices: [1, 2, 3]
  # Initial threshold for "scalar_threshold".
  nlsd_init_threshold:
    type: float
    low: 0.1
    high: 10.0
    log: true
  # Reference threshold_lr / alpha_lr: dedicated learning rates for the
  # nonlinearity threshold and the per-stalk diffusion gate.
  threshold_lr:
    type: float
    low: 0.0001
    high: 0.1
    log: true
  alpha_lr:
    type: float
    low: 0.0001
    high: 0.1
    log: true
{extra_block}
config:
  n_trials: 250
  n_seeds_per_trial: 10  # Bodnar protocol: every trial trains all 10 folds
  std_weight: 0.0            # penalty weight on std; 0 = optimise pure mean
  study_name: nlsd-{dataset}-{variant}
  seed: 42
  storage: sqlite:///nlsd_{dataset}_{variant}.db   # for distributed sweeps
  wandb_project: sheaf-mpnn-runs       # uncomment to log to Weights & Biases
  wandb_entity: adavit
"""


def main() -> None:
    """Write one NLSD sweep YAML per (dataset, variant) pair."""
    configs = Path(__file__).resolve().parent.parent / "configs"
    # NLSD targets the undirected node-classification benchmarks: every
    # dataset that already carries an NSD sweep gets the NLSD grid.
    datasets = sorted(p.parent.name for p in configs.glob("*/nsd"))
    for dataset in datasets:
        out_dir = configs / dataset / "nlsd"
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            if variant == "orthogonal":
                extra_block = ORTH_BLOCK
            elif variant == "low_rank":
                extra_block = RANK_BLOCK
            else:
                extra_block = ""
            (out_dir / f"{variant}.yaml").write_text(
                TEMPLATE.format(
                    dataset=dataset, variant=variant, extra_block=extra_block
                )
            )
            print(f"wrote configs/{dataset}/nlsd/{variant}.yaml")


if __name__ == "__main__":
    main()
