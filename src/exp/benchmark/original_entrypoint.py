"""Standalone entrypoint executed by the ORIGINAL repo's own Python
interpreter (see scripts/benchmark/setup_original_env.sh) -- deliberately
does NOT import anything from `exp.*`, since it runs in a separate
venv/interpreter with none of this repo's packages installed.

Reimplements the original repo's per-epoch train/test loop
(twitter-research/neural-sheaf-diffusion's `exp/run.py::train`/`test`,
~15 lines each) directly against its model classes and dataset loaders,
instead of importing `exp.run` there: that module unconditionally calls
`wandb.log()` inside the loop and only supports running all folds in one
process starting at fold 0. Reusing `models.disc_models` and
`utils.heterophilic` instead avoids both problems and needs no
wandb/GitPython dependency in the original repo's env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

VARIANT_TO_CLASS = {
    "DiagSheaf": "DiscreteDiagSheafDiffusion",
    "BundleSheaf": "DiscreteBundleSheafDiffusion",
    "GeneralSheaf": "DiscreteGeneralSheafDiffusion",
}

# Mirrors the defaults declared in src/exp/benchmark/hparams.py's
# OriginalHParams. Duplicated deliberately: this module runs under the
# vendored repo's own Python 3.9 interpreter and must not import `exp.*`.
DEFAULT_ARGS: dict[str, Any] = {
    "epochs": 1500,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "sheaf_decay": None,
    "early_stopping": 200,
    "stop_strategy": "loss",
    "input_dropout": 0.0,
    "dropout": 0.0,
    "normalised": True,
    "deg_normalised": False,
    "linear": False,
    "left_weights": True,
    "right_weights": True,
    "add_lp": False,
    "add_hp": False,
    "use_act": True,
    "second_linear": False,
    "orth": "householder",
    "sheaf_act": "tanh",
    "edge_weights": True,
    "sparse_learner": False,
    "max_t": 1.0,
}


def write_result_atomically(path: Path, result: dict[str, Any]) -> None:
    """Write *result* as JSON so readers never observe a partial file.

    Slurm can preempt this process mid-write; a truncated JSON at the final
    path would be treated as a completed job forever by ``pending_jobs``.
    Writing to a sibling temp file and ``os.replace``-ing it into place makes
    the result appear atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, path)


def _steady_state_epoch_time(epoch_times: list[float], warmup_epochs: int) -> float:
    steady = epoch_times[warmup_epochs:] or epoch_times
    return sum(steady) / len(steady) if steady else 0.0


def run_fold(
    *,
    model_cls: type,
    torch_module: Any,
    dataset: Any,
    data: Any,
    hparams: dict[str, Any],
    mode: str,
    epoch_cap: int,
    warmup_epochs: int,
    device: str,
) -> dict[str, Any]:
    """Train+evaluate one fold/seed of an original-repo discrete model, with
    the same wall-clock/peak-memory instrumentation used on the "ours" side.
    ``torch_module`` is the real ``torch`` module, injected so this function
    has no import-time dependency on the vendored repo.
    """
    F = torch_module.nn.functional
    args: dict[str, Any] = {**DEFAULT_ARGS, **hparams, "device": device}
    args["graph_size"] = data.x.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    sheaf_decay = (
        args["sheaf_decay"] if args["sheaf_decay"] is not None else args["weight_decay"]
    )

    model = model_cls(data.edge_index, args).to(device)
    sheaf_params, other_params = model.grouped_parameters()
    optimizer = torch_module.optim.Adam(
        [
            {"params": sheaf_params, "weight_decay": sheaf_decay},
            {"params": other_params, "weight_decay": args["weight_decay"]},
        ],
        lr=args["lr"],
    )

    def train_epoch() -> None:
        model.train()
        optimizer.zero_grad()
        out = model(data.x)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    def evaluate() -> tuple[float, float]:
        model.eval()
        with torch_module.no_grad():
            logits = model(data.x)
            val_pred = logits[data.val_mask].max(1)[1]
            val_acc = (
                val_pred.eq(data.y[data.val_mask]).sum().item()
                / data.val_mask.sum().item()
            )
            val_loss = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
            test_pred = logits[data.test_mask].max(1)[1]
            test_acc = (
                test_pred.eq(data.y[data.test_mask]).sum().item()
                / data.test_mask.sum().item()
            )
        score = val_acc if args["stop_strategy"] == "acc" else -val_loss
        return score, test_acc

    # Gate on the target device, not on CUDA availability: a CPU run on a GPU
    # host must not query CUDA memory stats for device "cpu".
    on_cuda = (
        torch_module.cuda.is_available() and torch_module.device(device).type == "cuda"
    )
    if on_cuda:
        torch_module.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    epoch_times: list[float] = []
    best_score, test_at_best, bad_counter, epoch = float("-inf"), 0.0, 0, 0
    max_epochs = epoch_cap if mode == "efficiency" else args["epochs"]
    patience = max_epochs if mode == "efficiency" else args["early_stopping"]

    for epoch in range(max_epochs):  # noqa: B007 -- used below for epochs_trained
        epoch_start = time.perf_counter()
        train_epoch()
        epoch_times.append(time.perf_counter() - epoch_start)

        if mode == "convergence":
            score, test_acc = evaluate()
            if score > best_score:
                best_score, test_at_best, bad_counter = score, test_acc, 0
            else:
                bad_counter += 1
            if bad_counter >= patience:
                break

    fit_time_s = time.perf_counter() - start
    epochs_trained = epoch + 1
    peak_gpu_mem_mb = (
        torch_module.cuda.max_memory_allocated(device) / 2**20 if on_cuda else None
    )

    inference_latency_ms = None
    if mode == "efficiency":
        # Efficiency mode never calls evaluate() inside the timed loop above
        # (that would double-count validation/test cost into fit_time_s, see
        # the module docstring's sibling discussion in ours_entrypoint.py).
        # A single untimed post-loop pass gives test_at_best a real value for
        # the sanity-only test_metric field instead of leaving it at 0.0.
        _, test_at_best = evaluate()

        model.eval()
        with torch_module.no_grad():
            if on_cuda:
                torch_module.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(data.x)
            if on_cuda:
                torch_module.cuda.synchronize(device)
            inference_latency_ms = (time.perf_counter() - t0) * 1000

    epoch_time_s = (
        _steady_state_epoch_time(epoch_times, warmup_epochs)
        if mode == "efficiency"
        else fit_time_s / max(epochs_trained, 1)
    )

    return {
        "fit_time_s": fit_time_s,
        "epoch_time_s": epoch_time_s,
        "epochs_trained": epochs_trained,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "param_count": sum(p.numel() for p in model.parameters()),
        "inference_latency_ms": inference_latency_ms,
        "test_metric": test_at_best,
        "metric_name": "acc",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_TO_CLASS))
    parser.add_argument(
        "--variant-label",
        default=None,
        help=(
            "Canonical (ours-side) variant name recorded in the result JSON, "
            "e.g. 'orthogonal'. Only affects the output record; --variant still "
            "selects the original repo's model class. Defaults to --variant."
        ),
    )
    parser.add_argument("--mode", required=True, choices=["convergence", "efficiency"])
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cuda", type=int, default=0)
    # Same default as ours_entrypoint._Args.base_seed: neither side's seeds
    # need to match the other's (pairing is by fold/index, and the fixed data
    # splits already match), but a differing default reads as an accident.
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epoch-cap", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, args.repo_path)
    import random

    import numpy as np
    import torch
    import yaml
    from models.disc_models import (  # ty: ignore[unresolved-import]
        DiscreteBundleSheafDiffusion,
        DiscreteDiagSheafDiffusion,
        DiscreteGeneralSheafDiffusion,
    )
    from utils.heterophilic import (  # ty: ignore[unresolved-import]
        get_dataset,
        get_fixed_splits,
    )

    model_cls = {
        "DiagSheaf": DiscreteDiagSheafDiffusion,
        "BundleSheaf": DiscreteBundleSheafDiffusion,
        "GeneralSheaf": DiscreteGeneralSheafDiffusion,
    }[args.variant]

    fold = args.index if args.mode == "convergence" else 0
    seed = args.base_seed + args.index
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    dataset = get_dataset(args.dataset)
    data = get_fixed_splits(dataset[0], args.dataset, fold).to(device)
    hparams = yaml.safe_load(Path(args.hparams).read_text())

    result = run_fold(
        model_cls=model_cls,
        torch_module=torch,
        dataset=dataset,
        data=data,
        hparams=hparams,
        mode=args.mode,
        epoch_cap=args.epoch_cap,
        warmup_epochs=args.warmup_epochs,
        device=device,
    )
    result.update(
        dataset=args.dataset,
        variant=args.variant_label or args.variant,
        side="original",
        mode=args.mode,
        index=args.index,
    )

    write_result_atomically(Path(args.out), result)


if __name__ == "__main__":
    main()
