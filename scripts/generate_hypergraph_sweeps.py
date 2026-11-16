"""Generate eight independent 250-trial studies per benchmark."""

from pathlib import Path

import yaml

from exp.datasets.hypergraph import HYPERGRAPHS
from exp.sweeps.models import SweepConfig


def generate(root=Path("configs/hypergraphs")):
    for dataset in HYPERGRAPHS:
        for model in ("hsnn", "dhsnn"):
            for variant in ("diagonal", "orthogonal", "low_rank", "general"):
                study = f"{dataset}-{model}-{variant}"
                space = {
                    "variant": {"type": "categorical", "choices": [variant]},
                    "stalk_dim": {"type": "categorical", "choices": [2, 4, 8]},
                    "hidden_dim": {"type": "categorical", "choices": [16, 32, 64]},
                    "num_layers": {"type": "int", "low": 1, "high": 6},
                    "input_dropout": {"type": "float", "low": 0.0, "high": 0.8},
                    "dropout": {"type": "categorical", "choices": [0.0, 0.2, 0.5]},
                    "lr": {"type": "float", "low": 0.001, "high": 0.02, "log": True},
                    "weight_decay": {
                        "type": "categorical",
                        "choices": [0.0, 0.00005, 0.0005],
                    },
                }
                if model == "dhsnn":
                    space["dhsnn_q"] = {
                        "type": "categorical",
                        "choices": [0.0, 0.125, 0.25, 0.5],
                    }
                    space.update(
                        {
                            "dhsnn_dynamic_sheaf": {
                                "type": "categorical",
                                "choices": [False, True],
                            },
                            "dhsnn_prediction": {
                                "type": "categorical",
                                "choices": ["MLP_var1", "MLP_var2"],
                            },
                            "dhsnn_map_activation": {
                                "type": "categorical",
                                "choices": ["sigmoid", "tanh"],
                            },
                            "dhsnn_left_projection": {
                                "type": "categorical",
                                "choices": [False, True],
                            },
                            "dhsnn_classifier_hidden": {
                                "type": "categorical",
                                "choices": [32, 64, 128],
                            },
                        }
                    )
                if variant == "orthogonal":
                    space["orth_strategy"] = {
                        "type": "categorical",
                        "choices": ["cayley", "householder"],
                    }
                if variant == "low_rank":
                    space["rank"] = {"type": "categorical", "choices": [1, 2]}
                spec = {
                    "model": model,
                    "dataset": {
                        "name": dataset,
                        "root": "data/cora-dshn"
                        if dataset == "cora-hypergraph"
                        else "data/hypergraphs",
                        "directed": model == "dhsnn",
                    },
                    "search_space": space,
                    "config": {
                        "n_trials": 250,
                        "budget_states": "finished",
                        "n_seeds_per_trial": 10,
                        "seed": 0,
                        "cuda": 0,
                        "study_name": study,
                        "storage": f"sqlite:///runs/hypergraph-sweeps/{study}.db",
                    },
                }
                SweepConfig.model_validate(spec)
                folder = root / dataset / model
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{variant}.yaml").write_text(
                    f"# sheaf sweep --yaml-path {folder.as_posix()}/{variant}.yaml "
                    f"--preset {dataset}\n" + yaml.safe_dump(spec, sort_keys=False),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    generate()
