# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Lio

"""Shared full-batch HSNN/DHSNN runner on audited frozen hypergraph splits."""

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.version
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.metrics import precision_score, recall_score

from exp.config import Config, ModelConfig, ModelType, RegConfig
from exp.datasets.hypergraph import (
    artifact_path,
    ensure_hypergraph,
    is_hypergraph,
    load_artifact,
    load_split_bank,
)
from exp.registries.models import model_registry
from sheaf_mpnn import HSNNVariant

_console = Console()


def evaluation_metric(args) -> str:
    """Use the reference Molecular F1 for hyperedge classification."""
    return "f1" if getattr(args, "task", "node") == "hyperedge" else "accuracy"


def _display_startup(args, x, y, index, num_edges, splits) -> None:
    """Match the graph runner's panel while naming hypergraph quantities."""
    n_train, n_val, n_test = (part.numel() for part in splits[0])
    test_status = "enabled (best checkpoint)" if args.evaluate_test else "disabled"
    content = (
        f"[bold]Dataset[/bold]  {getattr(args, 'dataset_name', 'cora_hypergraph')}  "
        f"N={x.size(0):,}  E={num_edges:,} (hyperedges)  "
        f"I={index.size(1):,}\n"
        f"         F={x.size(1)}  C={int(y.max()) + 1}  "
        f"metric={evaluation_metric(args)}  "
        f"avg_deg={index.size(1) / x.size(0):.1f}\n"
        f"[bold]Split[/bold]    train={n_train:,}  val={n_val:,}  test={n_test:,}  "
        f"({args.runs} frozen splits)\n"
        f"[bold]Model[/bold]    {getattr(args, 'model', 'hsnn').upper()}-"
        f"{args.variant}  d={args.stalk_dim}  "
        f"hidden={args.hidden_dim}  layers={args.num_layers}  "
        f"dropout={args.dropout}\n"
        f"[bold]Training[/bold] epochs={args.epochs}  patience={args.patience}  "
        f"monitor=val_{evaluation_metric(args)}  device={args.device}\n"
        f"[bold]Test[/bold]     {test_status}"
    )
    _console.print(
        Panel(
            content, title=f"{getattr(args, 'model', 'hsnn').upper()} Cross-Validation"
        )
    )


def _display_results(args, results: list[dict]) -> None:
    """Use the graph runner's percentage table and population standard deviation."""
    metric = evaluation_metric(args)
    label = ("Test " if args.evaluate_test else "Val ") + metric
    table = Table(
        title=(
            f"{getattr(args, 'model', 'hsnn')}-{args.variant} "
            f"on {getattr(args, 'dataset_name', 'cora_hypergraph')} "
            f"[{len(results)} splits]"
        )
    )
    table.add_column("Fold", justify="right", style="cyan")
    table.add_column(label, justify="right")
    values = []
    for fold, result in enumerate(results):
        value = (
            result["test"][metric]
            if args.evaluate_test
            else result[f"best_val_{metric}"]
        )
        values.append(value)
        table.add_row(str(fold), f"{value * 100:.4f}%")
    arr = np.array(values)
    table.add_section()
    table.add_row(
        "Mean +- Std",
        f"[bold]{arr.mean() * 100:.2f} +- {arr.std() * 100:.2f}%[/bold]",
    )
    _console.print(table)


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load a numeric incidence dataset and reject malformed inputs."""
    with np.load(path, allow_pickle=False) as data:
        if "dataset_id" in data and data["dataset_id"].item() == "dshn-hypergraph-v1":
            return load_artifact(path)
        if "dataset_id" not in data or data["dataset_id"].item() != "dshn-cora-q0-v1":
            raise ValueError(
                "Unverified/legacy Cora topology. Run scripts/prepare_hsnn_cora.py "
                "and use cora-dshn-q0.npz (1565 edges), not cora-cocitation.npz."
            )
        x = torch.from_numpy(data["x"].copy())
        y = torch.from_numpy(data["y"].copy())
        index = torch.from_numpy(data["hyperedge_index"].copy())
        num_edges = int(data["num_hyperedges"].item())
    if x.ndim != 2 or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError("x must be a finite float32 feature matrix")
    if y.shape != (x.size(0),) or y.dtype != torch.long:
        raise ValueError("y must be an int64 label vector matching x")
    if not torch.equal(y.unique(), torch.arange(int(y.max()) + 1)):
        raise ValueError("labels must be contiguous nonnegative class IDs")
    if index.dtype != torch.long or index.ndim != 2 or index.size(0) != 2:
        raise ValueError("hyperedge_index must be int64 with shape [2, K]")
    if num_edges < 1 or index.numel() == 0:
        raise ValueError("a nonempty hypergraph is required")
    if (index < 0).any() or int(index[0].max()) >= x.size(0):
        raise ValueError("invalid node index")
    if int(index[1].max()) >= num_edges:
        raise ValueError("invalid hyperedge index")
    if index.unique(dim=1).size(1) != index.size(1):
        raise ValueError("duplicate incidences")
    if int(torch.bincount(index[1], minlength=num_edges).max()) <= 2:
        raise ValueError("expected genuine higher-order hyperedges")
    if (x.shape, y.unique().numel(), num_edges, index.size(1)) != (
        (2708, 1433),
        7,
        1565,
        6994,
    ):
        raise ValueError("dataset does not match the audited Cora reference statistics")
    return x, y, index, num_edges


def load_provenance(path: Path) -> dict:
    """Require provenance bound to both the data and the frozen split file."""
    metadata = json.loads(path.with_suffix(".provenance.json").read_text("utf-8"))
    if (
        metadata.get("dataset_id") not in ("dshn-cora-q0-v1", "dshn-hypergraph-v1")
        or metadata.get("reference_commit")
        != "d94f81bc7d86bd8d41e416083063f3446c1ad796"
    ):
        raise ValueError("unsupported Cora provenance")
    for file, key in (
        (path, "dataset_sha256"),
        (path.with_suffix(".splits.npz"), "splits_sha256"),
    ):
        if hashlib.sha256(file.read_bytes()).hexdigest() != metadata.get(key):
            raise ValueError(f"Artifact checksum mismatch: {file}")
    return metadata


def load_splits(
    path: Path, num_nodes: int, runs: int, dataset_sha256: str
) -> list[tuple[torch.Tensor, ...]]:
    """Require exhaustive, disjoint frozen splits bound to this exact dataset."""
    with np.load(path, allow_pickle=False) as data:
        if "num_splits" in data:
            return load_split_bank(path, num_nodes, runs, dataset_sha256)
        if data["dataset_sha256"].item() != dataset_sha256:
            raise ValueError("split file belongs to a different dataset")
        arrays = [data[name].copy() for name in ("train", "val", "test")]
    n_train = int(num_nodes * 0.5)
    n_val = int(num_nodes * 0.25)
    for array, width in zip(
        arrays, (n_train, n_val, num_nodes - n_train - n_val), strict=True
    ):
        if array.dtype != np.int64 or array.ndim != 2 or array.shape[1] != width:
            raise ValueError(
                "split arrays must be int64 [runs, subset_size] at 50/25/25"
            )
    counts = {array.shape[0] for array in arrays}
    if len(counts) != 1 or runs < 1 or runs > arrays[0].shape[0]:
        raise ValueError("requested runs exceed the shared split bank")
    for row in range(arrays[0].shape[0]):
        joined = np.concatenate([array[row] for array in arrays])
        if not np.array_equal(np.sort(joined), np.arange(num_nodes)):
            raise ValueError("split overlap, omission or out-of-range node")
    return [tuple(torch.from_numpy(array[i]) for array in arrays) for i in range(runs)]


def source_hashes() -> dict[str, str]:
    """Record the actual imported library and runner, including uncommitted edits."""
    import sheaf_mpnn

    package = Path(sheaf_mpnn.__file__).parent
    paths = {
        f"sheaf_mpnn/{p.relative_to(package).as_posix()}": p
        for p in package.rglob("*.py")
    }
    paths["exp/hsnn_run.py"] = Path(__file__)
    paths["exp/dhsnn_run.py"] = Path(__file__).with_name("dhsnn_run.py")
    for relative in (
        "config.py",
        "datasets/hypergraph.py",
        "run.py",
        "registries/models.py",
        "registries/presets.py",
    ):
        paths[f"exp/{relative}"] = Path(__file__).parent / relative
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }


def metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Return accuracy and the reference harmonic mean of macro precision/recall."""
    loss = F.cross_entropy(logits, y).item()
    if not math.isfinite(loss):
        raise RuntimeError("non-finite evaluation loss")
    accuracy = (logits.argmax(-1) == y).float().mean().item()
    target = y.detach().cpu().numpy()
    prediction = logits.argmax(-1).detach().cpu().numpy()
    precision = precision_score(target, prediction, average="macro", zero_division=0)
    recall = recall_score(target, prediction, average="macro", zero_division=0)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_run(
    args: argparse.Namespace,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    seed: int,
    subsets: tuple[torch.Tensor, ...],
    split_row: int,
    incidence_roles: torch.Tensor | None = None,
) -> dict:
    """Select the first strictly best validation score; test on request."""
    x, y, index, num_edges = data
    metric = evaluation_metric(args)
    torch.manual_seed(seed)
    if x.is_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(x.device)
    run_dir = args.output / f"run-{split_row:02d}-seed-{seed}"
    run_dir.mkdir()
    np.savez(
        run_dir / "split.npz",
        train=subsets[0].numpy(),
        val=subsets[1].numpy(),
        test=subsets[2].numpy(),
    )
    train, val, test = (subset.to(x.device) for subset in subsets)
    model_cfg = ModelConfig(
        type=ModelType(getattr(args, "model", "hsnn")),
        variant=args.variant,
        stalk_dim=args.stalk_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rank=getattr(args, "rank", 1),
        orth_strategy=getattr(args, "orth_strategy", "cayley"),
        hsnn_residual=args.residual,
        hsnn_degree_regularization=args.degree_regularization,
        dhsnn_q=getattr(args, "q", 0.25),
        dhsnn_activation=getattr(args, "activation", "complex_relu"),
    )
    if model_cfg.type == ModelType.DHSNN:
        for field in fields(ModelConfig):
            if field.name.startswith("dhsnn_") and hasattr(args, field.name):
                setattr(model_cfg, field.name, getattr(args, field.name))
    model = model_registry.build(
        model_cfg.type.value,
        x.size(1),
        int(y.max()) + 1,
        model_cfg,
        RegConfig(
            input_dropout=getattr(args, "input_dropout", 0.0), dropout=args.dropout
        ),
    ).to(x.device)
    forward_kwargs: dict[str, int | str | torch.Tensor] = {"num_hyperedges": num_edges}
    if incidence_roles is not None:
        forward_kwargs["incidence_roles"] = incidence_roles
    if getattr(args, "task", "node") == "hyperedge":
        forward_kwargs["task"] = "hyperedge"
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_score = -math.inf
    best_val_loss = math.inf
    best_epoch = 0
    best_state = None
    epoch_seconds = []
    started = time.monotonic()
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, index, **forward_kwargs)
            loss = F.cross_entropy(logits[train], y[train])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite training loss: seed={seed}, epoch={epoch}"
                )
            loss.backward()
            for name, parameter in model.named_parameters():
                light_predictor = (
                    getattr(model_cfg, "dhsnn_light", False)
                    and model_cfg.type == ModelType.DHSNN
                    and name.startswith("map_learners.")
                )
                if parameter.grad is None and light_predictor:
                    continue
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"missing or non-finite gradient: {name}")
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluated = model(x, index, **forward_kwargs)
                val_metrics = metrics(evaluated[val], y[val])
            if val_metrics[metric] > best_score:
                best_score = val_metrics[metric]
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            row = {
                "seed": seed,
                "epoch": epoch,
                "train_loss": loss.item(),
                "val": val_metrics,
                "best_epoch": best_epoch,
                "epoch_seconds": time.monotonic() - epoch_start,
            }
            epoch_seconds.append(row["epoch_seconds"])
            history.write(json.dumps(row) + "\n")
            history.flush()
            if epoch - best_epoch >= args.patience:
                break
    assert best_state is not None
    checkpoint = {"state_dict": best_state, "epoch": best_epoch, "seed": seed}
    torch.save(checkpoint, run_dir / "best.pt")
    restored = torch.load(run_dir / "best.pt", map_location=x.device, weights_only=True)
    model.load_state_dict(restored["state_dict"])
    model.eval()
    test_metrics = None
    if args.evaluate_test:
        with torch.no_grad():
            logits = model(x, index, **forward_kwargs)
            test_metrics = metrics(logits[test], y[test])
    result = {
        "seed": seed,
        "split_row": split_row,
        "epochs_ran": epoch,
        "best_epoch": best_epoch,
        "metric": metric,
        f"best_val_{metric}": best_score,
        "best_val_loss": best_val_loss,
        "test": test_metrics,
        "split_sizes": [int(part.numel()) for part in subsets],
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.monotonic() - started,
        "median_epoch_seconds": float(np.median(epoch_seconds)),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(x.device)
        if x.is_cuda
        else 0,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Run independent splits and save configuration, checkpoints and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/cora-dshn/cora-dshn-q0.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--seed", type=int, default=0, help="Model seed, independent of frozen splits"
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Enable only for a preselected final configuration",
    )
    parser.add_argument("--stalk-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--variant", choices=[v.name.lower() for v in HSNNVariant], default="diagonal"
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--degree-regularization", type=float, default=0.0)
    parser.add_argument(
        "--residual", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if min(args.runs, args.epochs, args.patience) < 1:
        parser.error("runs, epochs and patience must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    execute(args)


def execute(args: argparse.Namespace, cli_config: dict | None = None) -> dict:
    """Single training implementation used by the preset and legacy script."""
    if min(args.runs, args.epochs, args.patience) < 1:
        raise ValueError("runs, epochs and patience must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    torch.set_num_threads(2)
    provenance = load_provenance(args.data)
    from exp.datasets.hypergraph import validate_reference_artifact

    validate_reference_artifact(
        provenance.get("dataset_name", "cora-hypergraph"), args.data
    )
    x, y, index, num_edges = load_data(args.data)
    splits = load_splits(
        args.data.with_suffix(".splits.npz"),
        y.size(0),
        args.runs,
        provenance["dataset_sha256"],
    )
    args.task = provenance.get("task", "node")
    metric_name = evaluation_metric(args)
    roles = None
    if getattr(args, "model", "hsnn") == "dhsnn":
        from exp.dhsnn_run import load_roles

        roles = load_roles(args.data, index).to(args.device)
    sizes = torch.bincount(index[1], minlength=num_edges)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "cli_config": cli_config,
        "metric": metric_name,
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "dataset": {
            "name": provenance["dataset_id"],
            "nodes": x.size(0),
            "features": x.size(1),
            "classes": int(y.max()) + 1,
            "hyperedges": num_edges,
            "incidences": index.size(1),
            "max_edge_size": int(sizes.max()),
            "edges_larger_than_two": int((sizes > 2).sum()),
            "feature_normalization": "none",
            "added_singleton_edges": False,
            "incidence_roles_used": roles is not None,
            "tail_incidences": int((roles == -1).sum()) if roles is not None else None,
            "head_incidences": int((roles == 1).sum()) if roles is not None else None,
        },
        "protocol": (
            "transductive; frozen reference split bank (predefined or random "
            "according to dataset provenance); "
            f"first strictly best validation {metric_name}; optional final test"
        ),
        "data_provenance": provenance,
        "source_sha256": source_hashes(),
        "experiment_scope": (
            getattr(args, "model", "hsnn").upper()
            + " on audited reference topology; not tuned paper scores"
        ),
        "implementation": "dshn-message-passing-v2"
        if getattr(args, "model", "hsnn") == "dhsnn"
        else "hsnn-message-passing-v1",
        "reference_commit": "d94f81bc7d86bd8d41e416083063f3446c1ad796",
        "hyperparameter_selection": "fixed configuration; no Bayesian search",
        "seed_policy": (
            "splits: sequential seed-0 stream; model: seed+run "
            "(differs from reference's coupled RNG)"
        ),
        "bitwise_determinism_guaranteed": False,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_geometric": version("torch-geometric"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device)
        if str(args.device).startswith("cuda")
        else None,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    _display_startup(args, x, y, index, num_edges, splits)
    data = (x.to(args.device), y.to(args.device), index.to(args.device), num_edges)
    results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running folds...", total=args.runs)
        for fold in range(args.runs):
            result = train_run(args, data, args.seed + fold, splits[fold], fold, roles)
            results.append(result)
            if getattr(args, "model", "hsnn") == "dhsnn":
                _console.print(
                    f"Completed q={args.q} fold={fold} epochs={result['epochs_ran']} "
                    f"best_epoch={result['best_epoch']} "
                    f"val_{metric_name}={result[f'best_val_{metric_name}']:.6f}",
                    markup=False,
                )
            metric = (
                result["test"][metric_name]
                if args.evaluate_test
                else result[f"best_val_{metric_name}"]
            )
            label = ("test_" if args.evaluate_test else "val_") + metric_name
            progress.update(
                task, advance=1, description=f"fold {fold:2d}  {label}={metric:.4f}"
            )
    accuracy = np.array(
        [r["test"][metric_name] for r in results if r["test"] is not None]
    )
    validation = np.array([r[f"best_val_{metric_name}"] for r in results])
    summary = {
        "runs": results,
        "metric": metric_name,
        f"val_{metric_name}_mean": float(validation.mean()),
        "test_evaluated": args.evaluate_test,
        f"test_{metric_name}_mean": float(accuracy.mean()) if accuracy.size else None,
        f"test_{metric_name}_population_std": float(accuracy.std(ddof=0))
        if accuracy.size
        else None,
        f"test_{metric_name}_sample_std": float(accuracy.std(ddof=1))
        if accuracy.size > 1
        else None,
        "mean_run_seconds": float(np.mean([r["elapsed_seconds"] for r in results])),
        "max_peak_cuda_bytes": max(r["peak_cuda_bytes"] for r in results),
    }
    summary_path = args.output / "summary.json"
    temporary_summary = args.output / "summary.json.tmp"
    temporary_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary_summary.replace(summary_path)
    _display_results(args, results)
    _console.print(f"Results: {args.output / 'summary.json'}", markup=False)
    return summary


def run(cfg: Config) -> list[float]:
    """Run the audited hypergraph protocol from the common experiment config.

    Return test metrics by default, validation metrics when testing is disabled.
    Ordinary graph Lightning runs and sweeps do not implement this protocol.
    """
    if cfg.model.type != ModelType.HSNN or not is_hypergraph(cfg.dataset.name):
        raise ValueError("HSNN requires the hsnn_cora preset / cora_hypergraph dataset")
    if cfg.dataset.directed:
        raise ValueError(
            "HSNN is undirected; select model.type=dhsnn for directionality"
        )
    if cfg.optim.stop_strategy != "acc":
        raise ValueError("The audited HSNN protocol requires optim.stop_strategy=acc")
    if cfg.wandb.enabled:
        raise ValueError(
            "HSNN currently records local JSON artifacts; W&B is unsupported"
        )
    if cfg.cv.min_acc != 0:
        raise ValueError("HSNN does not abort based on held-out test accuracy")
    if (
        cfg.optim.sheaf_decay is not None
        or cfg.optim.threshold_lr is not None
        or cfg.optim.alpha_lr is not None
        or cfg.optim.batch_size != 1
    ):
        raise ValueError(
            "HSNN uses full-batch Adam with one learning rate/weight decay"
        )
    supported = {
        "type",
        "variant",
        "stalk_dim",
        "hidden_dim",
        "num_layers",
        "rank",
        "orth_strategy",
        "hsnn_residual",
        "hsnn_degree_regularization",
    }
    defaults = ModelConfig()
    unsupported = [
        f.name
        for f in fields(ModelConfig)
        if f.name not in supported
        and getattr(cfg.model, f.name) != getattr(defaults, f.name)
    ]
    if unsupported:
        raise ValueError(f"Unsupported HSNN model overrides: {unsupported}")
    output = (
        Path(cfg.hypergraph.output)
        if cfg.hypergraph.output
        else Path("runs")
        / (f"hsnn-{cfg.dataset.name}-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    )
    ensure_hypergraph(cfg.dataset.name, cfg.dataset.root)
    args = argparse.Namespace(
        data=artifact_path(cfg.dataset.name, cfg.dataset.root),
        dataset_name=cfg.dataset.name,
        output=output,
        device=f"cuda:{cfg.hardware.cuda}" if cfg.hardware.cuda >= 0 else "cpu",
        runs=cfg.cv.folds,
        seed=cfg.cv.seed,
        epochs=cfg.optim.epochs,
        patience=cfg.optim.early_stopping,
        evaluate_test=cfg.hypergraph.evaluate_test,
        stalk_dim=cfg.model.stalk_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        variant=cfg.model.variant,
        rank=cfg.model.rank,
        orth_strategy=cfg.model.orth_strategy,
        input_dropout=cfg.reg.input_dropout,
        dropout=cfg.reg.dropout,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        degree_regularization=cfg.model.hsnn_degree_regularization,
        residual=cfg.model.hsnn_residual,
    )
    summary = execute(args, cli_config=asdict(cfg))
    metric_name = summary.get("metric", "accuracy")
    return [
        r["test"][metric_name]
        if cfg.hypergraph.evaluate_test
        else r[f"best_val_{metric_name}"]
        for r in summary["runs"]
    ]
