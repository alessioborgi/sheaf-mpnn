"""Directed Cora experiments using the shared audited hypergraph protocol."""

import argparse
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from exp.config import Config, ModelConfig, ModelType
from exp.datasets.hypergraph import artifact_path, ensure_hypergraph, is_hypergraph
from exp.hsnn_run import execute
from sheaf_mpnn.directed_hypergraph_sheaf_conv import BaseDirectedHypergraphSheafConv


def load_roles(path: Path, index: torch.Tensor) -> torch.Tensor:
    """Require explicit tail/head labels aligned with every audited incidence."""
    with np.load(path, allow_pickle=False) as data:
        if "incidence_roles" not in data:
            raise ValueError("DHSNN requires incidence_roles")
        values = data["incidence_roles"].copy()
        if not np.issubdtype(values.dtype, np.signedinteger):
            raise ValueError("incidence_roles must be signed integers")
        if not np.isin(values, [-1, 1]).all():
            raise ValueError("incidence_roles must contain only -1 or +1")
        roles = torch.from_numpy(values.astype(np.int64))
    BaseDirectedHypergraphSheafConv._validate_roles(roles, index)
    return roles


def run(cfg: Config) -> list[float]:
    """Run the audited hypergraph protocol from the common experiment config.

    Return test metrics by default, validation metrics when testing is disabled.
    Ordinary graph Lightning runs and sweeps do not implement this protocol.
    """
    if cfg.model.type != ModelType.DHSNN or not is_hypergraph(cfg.dataset.name):
        raise ValueError(
            "DHSNN requires the dhsnn_cora preset / cora_hypergraph dataset"
        )
    if cfg.optim.stop_strategy != "acc":
        raise ValueError("The audited DHSNN protocol requires optim.stop_strategy=acc")
    if cfg.wandb.enabled:
        raise ValueError(
            "DHSNN currently records local JSON artifacts; W&B is unsupported"
        )
    if cfg.cv.min_acc != 0:
        raise ValueError("DHSNN does not abort based on held-out test accuracy")
    if (
        cfg.optim.sheaf_decay is not None
        or cfg.optim.threshold_lr is not None
        or cfg.optim.alpha_lr is not None
        or cfg.optim.batch_size != 1
    ):
        raise ValueError(
            "DHSNN uses full-batch Adam with one learning rate/weight decay"
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
        "dhsnn_q",
        "dhsnn_activation",
    }
    defaults = ModelConfig()
    supported.update(f.name for f in fields(ModelConfig) if f.name.startswith("dhsnn_"))
    unsupported = [
        f.name
        for f in fields(ModelConfig)
        if f.name not in supported
        and getattr(cfg.model, f.name) != getattr(defaults, f.name)
    ]
    if unsupported:
        raise ValueError(f"Unsupported DHSNN model overrides: {unsupported}")
    output = (
        Path(cfg.hypergraph.output)
        if cfg.hypergraph.output
        else Path("runs")
        / (f"dhsnn-{cfg.dataset.name}-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    )
    ensure_hypergraph(cfg.dataset.name, cfg.dataset.root)
    args = argparse.Namespace(
        model="dhsnn",
        q=cfg.model.dhsnn_q,
        activation=cfg.model.dhsnn_activation,
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
    for field in fields(ModelConfig):
        if field.name.startswith("dhsnn_"):
            setattr(args, field.name, getattr(cfg.model, field.name))
    summary = execute(args, cli_config=asdict(cfg))
    metric_name = summary.get("metric", "accuracy")
    return [
        r["test"][metric_name]
        if cfg.hypergraph.evaluate_test
        else r[f"best_val_{metric_name}"]
        for r in summary["runs"]
    ]
