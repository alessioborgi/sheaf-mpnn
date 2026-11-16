"""Translate pinned official DSHN search spaces into the library schema."""

from pathlib import Path

import yaml

from exp.datasets.hypergraph import HYPERGRAPHS
from exp.registries.presets import preset_registry
from exp.sweeps.models import SweepConfig

COMMIT = "d94f81bc7d86bd8d41e416083063f3446c1ad796"
FIELDS = {
    "AllSet_input_norm": "dhsnn_input_norm",
    "dropout": "dropout",
    "lr": "lr",
    "wd": "weight_decay",
    "sheaf_dropout": "dhsnn_sheaf_dropout",
    "MLP_hidden": "hidden_dim",
    "sheaf_num_layers": "num_layers",
    "d": "stalk_dim",
    "sheaf_act": "dhsnn_map_activation",
    "sheaf_left_proj": "dhsnn_left_projection",
    "residual": "hsnn_residual",
    "dynamic_sheaf": "dhsnn_dynamic_sheaf",
    "sheaf_special_head": "dhsnn_special_head",
    "add_identity": "dhsnn_add_identity",
    "sheaf_pred_block": "dhsnn_prediction",
    "Classifier_num_layers": "dhsnn_classifier_layers",
    "Classifier_hidden": "dhsnn_classifier_hidden",
    "q": "dhsnn_q",
    "epochs": "epochs",
    "early_stopping": "early_stopping",
    "light": "dhsnn_light",
}


def convert(parameters):
    """Preserve categorical values and inclusive integer bounds exactly."""
    result = {}
    for source, target in FIELDS.items():
        if source not in parameters:
            continue
        spec = parameters[source]
        if "min" in spec:
            result[target] = {"type": "int", "low": spec["min"], "high": spec["max"]}
        else:
            values = spec.get("values", [spec.get("value")])
            values = [
                True if v == "true" else False if v == "false" else v for v in values
            ]
            result[target] = {"type": "categorical", "choices": values}
    return result


def main():
    root = Path("tmp/dshn-source") / COMMIT / "configs/base"
    for light in (False, True):
        name = "SheafGeDiDiag" + ("Light" if light else "")
        source = yaml.safe_load((root / f"{name}.yaml").read_text())
        space = convert(source["parameters"])
        # User-selected training budget; upstream base YAML uses 500 epochs.
        space["epochs"] = {"type": "categorical", "choices": [1500]}
        space["variant"] = {"type": "categorical", "choices": ["diagonal"]}
        space["dhsnn_light"] = {"type": "categorical", "choices": [light]}
        for dataset in HYPERGRAPHS:
            cfg = preset_registry.get(dataset)
            study = f"paper1500-{dataset}-{name}"
            spec = {
                "model": "dhsnn",
                "dataset": {
                    "name": dataset,
                    "root": cfg.dataset.root,
                    "directed": True,
                },
                "search_space": space,
                "config": {
                    "n_trials": 250,
                    "n_seeds_per_trial": 10,
                    "std_weight": 0,
                    "seed": 0,
                    "study_name": study,
                    "storage": f"sqlite:///runs/paper-sweeps/{study}.db",
                },
            }
            SweepConfig.model_validate(spec)
            path = Path("configs/paper-dshn") / dataset / f"{name}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
