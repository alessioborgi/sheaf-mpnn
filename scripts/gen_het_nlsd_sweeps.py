# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Generate the het-NLSD sweep YAMLs under ``configs/<hgb_dataset>/het_nlsd/``.

One sweep per (HGB dataset, variant) pair. These are the first sweep configs
for a heterogeneous family, so they sweep the het context flags
(``use_node_types`` / ``use_edge_types`` / ``ensemble``) alongside the shared
grid and the NLSD-specific knobs.  ``num_node_types`` / ``num_edge_types`` are
deliberately *not* swept: they are properties of the loaded graph, back-filled
by ``HetSheafLightningModule`` from the dataset's ``HetDatasetInfo``.

Run from the repository root:

    uv run python scripts/gen_het_nlsd_sweeps.py
"""

from __future__ import annotations

from pathlib import Path

DATASETS = ("imdb", "dblp", "acm", "freebase")
VARIANTS = ("diagonal", "orthogonal", "general", "low_rank")

ORTH_BLOCK = """
  orth_strategy:
    type: categorical
    choices: [cayley, fasthpp, householder]
"""

RANK_BLOCK = """
  rank:
    type: categorical
    choices: [1, 2, 4]
"""

TEMPLATE = """\
# Het NLSD sweep on {dataset} - {variant} restriction map.
#
# Heterogeneous Nonlinear Sheaf Diffusion: the NLSD operator
# L^Phi = delta^T o Phi o delta (Zaghen et al., GRaM @ ICML 2024) with the
# restriction maps generated from the het context -- node features, node types,
# edge types, or a per-edge-type ensemble of generator MLPs.
#
# num_node_types / num_edge_types are not swept: they describe the loaded
# graph, and HetSheafLightningModule back-fills them from HetDatasetInfo.
#
# Run with:
#   python -m exp.sweeps.sweep --yaml-path configs/{dataset}/het_nlsd/{variant}.yaml \\
#     --preset {dataset}_het_nlsd_{variant}

model: het_nlsd

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
  learn_alpha:
    type: categorical
    choices: [false, true]
  jknet:
    type: categorical
    choices: [false, true]

  # Het context: which sources feed the restriction-map generator, and whether
  # each edge type gets its own generator MLP (TypeEnsemble) instead of one
  # shared MLP reading a one-hot edge type.
  use_node_types:
    type: categorical
    choices: [false, true]
  use_edge_types:
    type: categorical
    choices: [false, true]
  ensemble:
    type: categorical
    choices: [false, true]

  # Which Phi sits between the coboundary and its transpose. "identity" is
  # linear het NSD, which the het_nsd family already covers.
  nlsd_nonlinearity:
    type: categorical
    choices: [mlp, scalar_threshold, edge_threshold]
  nlsd_psi:
    type: categorical
    choices: [linear, cos]
  nlsd_mlp_layers:
    type: categorical
    choices: [1, 2, 3]
  nlsd_init_threshold:
    type: float
    low: 0.1
    high: 10.0
    log: true
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
  n_seeds_per_trial: 5
  std_weight: 0.0            # penalty weight on std; 0 = optimise pure mean
  study_name: het-nlsd-{dataset}-{variant}
  seed: 42
  storage: sqlite:///het_nlsd_{dataset}_{variant}.db   # for distributed sweeps
  wandb_project: sheaf-mpnn-runs       # uncomment to log to Weights & Biases
  wandb_entity: adavit
"""


def main() -> None:
    """Write one het-NLSD sweep YAML per (dataset, variant) pair."""
    configs = Path(__file__).resolve().parent.parent / "configs"
    for dataset in DATASETS:
        out_dir = configs / dataset / "het_nlsd"
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
            print(f"wrote configs/{dataset}/het_nlsd/{variant}.yaml")


if __name__ == "__main__":
    main()
