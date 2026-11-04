# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Generate the BuNN sweep YAMLs under ``configs/<dataset>/bunn/``.

BuNN has no restriction-map variant grid -- a flat bundle assigns one
orthogonal map per node -- so unlike the sheaf families it gets one sweep per
dataset, split by heat-kernel solver instead of by map family. Run from the
repository root:

    uv run python scripts/gen_bunn_sweeps.py
"""

from __future__ import annotations

from pathlib import Path

from sheaf_mpnn.bunn.bunn_layers import MAX_DENSE_EIGH_NODES

# Node counts for the benchmarks that carry an NSD sweep. Used only to decide
# which datasets can host a spectral sweep at all: that solver needs a dense
# [N, N] eigendecomposition, so above MAX_DENSE_EIGH_NODES every trial would
# fail to build and the sweep would be pure wasted compute. Taylor is sparse
# and is generated everywhere.
NUM_NODES = {
    "amazon_ratings": 24_492,
    "chameleon": 2_277,
    "chameleon_filtered": 890,
    "citeseer": 3_327,
    "cora": 2_708,
    "cornell": 183,
    "film": 7_600,
    "minesweeper": 10_000,
    "pubmed": 19_717,
    "questions": 48_921,
    "roman_empire": 22_662,
    "squirrel": 5_201,
    "squirrel_filtered": 2_223,
    "texas": 183,
    "tolokers": 11_758,
    "wisconsin": 251,
}

# The two evaluations of the same layer. They are swept separately because
# their usable diffusion times do not overlap: the truncated Taylor series is
# only defined for small t (past t*lambda_max ~ 8 its partial sums are pure
# cancellation error, and the layer refuses to build), while the spectral
# solver is exact at any t but needs a dense eigendecomposition.
SOLVERS = ("taylor", "spectral")

# Small-t regime. bunn_max_degree is deliberately not swept: it is an accuracy
# knob rather than a hyperparameter, and leaving it at 0 derives the smallest
# degree that is accurate for whichever bunn_tau a trial draws. Sweeping the
# two independently would propose pairs the layer rejects (t = 4 needs degree
# 32) and lose trials to configuration errors.
TAYLOR_BLOCK = """
  # Pinned: the preset carries the other solver's default, and the two are
  # swept separately because their usable diffusion times do not overlap.
  bunn_solver:
    type: categorical
    choices: [taylor]
  bunn_tau:
    type: float
    low: 0.05
    high: 4.0
    log: true
"""

# Large-t regime: this is where BuNN's long-range claim lives (the paper's best
# roman-empire configuration uses t = 100). bunn_num_eigs truncates the
# spectrum -- 0 keeps it whole and exact.
SPECTRAL_BLOCK = """
  bunn_solver:
    type: categorical
    choices: [spectral]
  bunn_tau:
    type: categorical
    choices: [0.1, 1.0, 10.0, 100.0]
  bunn_learn_tau:
    type: categorical
    choices: [false, true]
  bunn_num_eigs:
    type: categorical
    choices: [0, 32, 128]
"""
# The largest bunn_num_eigs choice above; a truncation cannot ask for more
# eigenpairs than the graph has nodes.
MAX_NUM_EIGS_CHOICE = 128

TEMPLATE = """\
# BuNN sweep on {dataset} - {solver} heat kernel.
#
# Bundle Neural Networks (Bamberger et al., ICLR 2025): message *diffusion* on
# a learned flat vector bundle. Each layer synchronises node features with a
# per-node orthogonal map, runs the graph heat kernel for a time t, and
# desynchronises (Lemma 3.1). A single layer couples every pair of nodes, so t
# -- not depth -- sets the scale at which distant nodes interact.
#
# stalk_dim is the bundle dimension d (2 in the paper's experiments) and
# bunn_num_bundles is how many run in parallel, so the node state is
# bunn_num_bundles * stalk_dim * hidden_dim wide. Keep that product in mind
# when reading the width choices below.
#
# Run with:
#   python -m exp.sweeps.sweep --yaml-path configs/{dataset}/bunn/{solver}.yaml \\
#     --preset {dataset}_bunn

model: bunn

search_space:
  # d = 2 is the paper's choice throughout: O(2) needs one angle per bundle,
  # and higher dimensions cost more without helping.
  stalk_dim:
    type: categorical
    choices: [2, 4]
  bunn_num_bundles:
    type: categorical
    choices: [2, 4, 8, 16, 32]
  hidden_dim:
    type: categorical
    choices: [4, 8, 16, 32]
  num_layers:
    type: int
    low: 1
    high: 8
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
  # Separate decay for phi, the per-node bundle learner -- BuNN's analogue of
  # the sheaf families' per-edge map generator.
  sheaf_decay:
    type: float
    low: 0.00000001
    high: 0.01
    log: true
{solver_block}
  # phi from Eq. 1. The paper allows any GNN here (unlike NSD, which restricts
  # it to an MLP) and reports that reading the graph structure helped.
  bunn_phi:
    type: categorical
    choices: [mlp, sum_gnn]
  bunn_phi_layers:
    type: int
    low: 1
    high: 4
  bunn_shared_phi:
    type: categorical
    choices: [false, true]
  # Laplacian driving the diffusion. The spectral solver requires "sym".
  bunn_normalize:
    type: categorical
    choices: [{normalize_choices}]
  # Only read when stalk_dim != 2; d = 2 uses the direct O(2) angle
  # parameterisation instead.
  orth_strategy:
    type: categorical
    choices: [cayley, fasthpp, householder]

config:
  n_trials: 250
  n_seeds_per_trial: 10  # Bodnar protocol: every trial trains all 10 folds
  std_weight: 0.0            # penalty weight on std; 0 = optimise pure mean
  study_name: bunn-{dataset}-{solver}
  seed: 42
  storage: sqlite:///bunn_{dataset}_{solver}.db   # for distributed sweeps
  wandb_project: sheaf-mpnn-runs       # uncomment to log to Weights & Biases
  wandb_entity: adavit
"""


def main() -> None:
    """Write one BuNN sweep YAML per (dataset, solver) pair."""
    configs = Path(__file__).resolve().parent.parent / "configs"
    # BuNN targets the undirected node-classification benchmarks: every dataset
    # that already carries an NSD sweep gets the BuNN grid.
    datasets = sorted(p.parent.name for p in configs.glob("*/nsd"))
    for dataset in datasets:
        out_dir = configs / dataset / "bunn"
        out_dir.mkdir(parents=True, exist_ok=True)
        num_nodes = NUM_NODES.get(dataset)
        for solver in SOLVERS:
            taylor = solver == "taylor"
            if not taylor and (
                num_nodes is None
                or num_nodes > MAX_DENSE_EIGH_NODES
                or num_nodes < MAX_NUM_EIGS_CHOICE
            ):
                print(
                    f"skipping configs/{dataset}/bunn/spectral.yaml: "
                    f"{num_nodes} nodes exceeds the dense-eigh limit "
                    f"({MAX_DENSE_EIGH_NODES})"
                )
                continue
            (out_dir / f"{solver}.yaml").write_text(
                TEMPLATE.format(
                    dataset=dataset,
                    solver=solver,
                    solver_block=TAYLOR_BLOCK if taylor else SPECTRAL_BLOCK,
                    # The rw Laplacian is not symmetric, so its eigenvectors are
                    # not orthonormal; the spectral solver rejects it.
                    normalize_choices="sym, rw" if taylor else "sym",
                )
            )
            print(f"wrote configs/{dataset}/bunn/{solver}.yaml")


if __name__ == "__main__":
    main()
