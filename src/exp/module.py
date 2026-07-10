# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""PyTorch Lightning module wrapping Sheaf models.

This module provides the `SheafLightningModule`, which handles the training loop,
evaluation metrics, and optimizer configuration for all sheaf-based models.
It is designed to be compatible with both transductive (node classification)
and inductive (graph classification) tasks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from exp.config import Config
from exp.data import DatasetInfo
from exp.registries.models import model_registry


class SheafLightningModule(LightningModule):
    """Wraps Sheaf models with Lightning training / evaluation logic.

    This class serves as the interface between the raw model and the PyTorch
    Lightning Trainer. It handles loss calculation, metric tracking (ACC/AUC),
    and hardware-agnostic execution.

    Args:
        cfg: Global configuration object containing model and optimization params.
        info: Metadata about the dataset (num_features, num_classes, metric, etc.).
    """

    def __init__(self, cfg: Config, info: DatasetInfo) -> None:
        super().__init__()
        self.cfg = cfg
        self.info = info
        try:
            # Dynamically instantiate the model from the registry
            self.model: Any = model_registry.build(
                str(cfg.model.type),
                info.num_features,
                info.num_classes,
                cfg.model,
                cfg.reg,
            )
        except KeyError as exc:
            raise ValueError(f"Unknown model type: {cfg.model.type!r}") from exc

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def _forward(self, data) -> torch.Tensor:
        """Shared forward pass helper. Subclasses override this for het support."""
        return self.model(data.x, data.edge_index)

    def training_step(self, batch, batch_idx):
        """Standard training step: forward pass + cross-entropy loss."""
        data = batch
        # Forward pass through the sheaf model
        logits = self._forward(data)

        # Calculate loss only on the training subset
        if getattr(self.info, "task", "multiclass") == "multilabel":
            loss = F.binary_cross_entropy_with_logits(
                logits[data.train_mask], data.y[data.train_mask].float()
            )
        else:
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Validation step using the shared evaluation logic."""
        self._eval_step(batch, "val_mask", "val")

    def test_step(self, batch, batch_idx) -> None:
        """Test step using the shared evaluation logic."""
        self._eval_step(batch, "test_mask", "test")

    # ------------------------------------------------------------------
    # Shared evaluation logic
    # ------------------------------------------------------------------

    def _eval_step(self, data, mask_attr: str, prefix: str) -> None:
        """Common evaluation logic for validation and testing.

        Args:
            data: The batch object (Data or HeteroData).
            mask_attr: The name of the mask attribute (e.g., 'val_mask').
            prefix: Metric prefix for logging (e.g., 'val').
        """
        # Forward pass (Dropout is disabled automatically by Lightning)
        logits = self._forward(data)

        # Handle numerical instability (Inf/NaN) gracefully
        prog = prefix == "val"  # Show validation metrics in the progress bar
        if not torch.isfinite(logits).all():
            bad = 0.0 if self.info.metric == "acc" else 0.5
            self.log(
                f"{prefix}_loss",
                torch.tensor(float("inf")),
                on_step=False,
                on_epoch=True,
                prog_bar=prog,
                batch_size=1,
            )
            self.log(
                f"{prefix}_{self.info.metric}",
                torch.tensor(bad),
                on_step=False,
                on_epoch=True,
                prog_bar=prog,
                batch_size=1,
            )
            return

        # Filter predictions and ground truth based on the provided mask
        mask = getattr(data, mask_attr)
        if getattr(self.info, "task", "multiclass") == "multilabel":
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], data.y[mask].float()
            )
        else:
            loss = F.cross_entropy(logits[mask], data.y[mask])
        metric = self._compute_metric(logits, data.y, mask)

        # Log results
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )
        self.log(
            f"{prefix}_{self.info.metric}",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )
        self._log_extra_metrics(logits, data.y, mask, prefix)

    def _log_extra_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        prefix: str,
    ) -> None:
        probs = F.softmax(logits[mask], dim=-1).detach().cpu().numpy()
        y_true = labels[mask].detach().cpu().numpy()
        y_pred = probs.argmax(axis=-1)
        n_classes = probs.shape[1]

        kw: dict = {"on_step": False, "on_epoch": True, "batch_size": 1}
        self.log(
            f"{prefix}_f1_macro",
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            **kw,
        )
        self.log(
            f"{prefix}_f1_micro",
            float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            **kw,
        )
        self.log(
            f"{prefix}_precision_macro",
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            **kw,
        )
        self.log(
            f"{prefix}_recall_macro",
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            **kw,
        )

        present = np.unique(y_true)
        if present.size < 2:
            return

        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
            pr_auc = float(average_precision_score(y_true, probs[:, 1]))
        else:
            # OVR AUC/AP are undefined for classes absent from the split;
            # restrict to present classes (identical when all are present).
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            y_bin, probs_present = y_bin[:, present], probs[:, present]
            roc_auc = float(roc_auc_score(y_bin, probs_present, average="macro"))
            pr_auc = float(
                average_precision_score(y_bin, probs_present, average="macro")
            )

        # On roc_auc datasets _eval_step already logs this key as the primary
        # metric; Lightning forbids re-logging it with different arguments.
        if self.info.metric != "roc_auc":
            self.log(f"{prefix}_roc_auc", roc_auc, **kw)
        self.log(f"{prefix}_pr_auc", pr_auc, **kw)

    def _compute_metric(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Calculate Accuracy or ROC-AUC depending on the dataset requirements.

        Args:
            logits: Unnormalized model outputs.
            labels: Ground truth class indices.
            mask: Boolean mask identifying nodes/graphs in the current split.
        """
        # Case 1: Accuracy (multiclass or multilabel)
        if self.info.metric == "acc":
            if getattr(self.info, "task", "multiclass") == "multilabel":
                pred = (torch.sigmoid(logits[mask]) > 0.5).float()
                correct = (pred == labels[mask].float()).all(dim=-1).sum()
                return float(correct.item()) / int(mask.sum().item())
            pred = logits[mask].argmax(dim=-1)
            return float(pred.eq(labels[mask]).sum().item()) / int(mask.sum().item())

        # Case 2: ROC-AUC (used for heterophilous/binary tasks)
        # AUC is computed on CPU probabilities for sklearn compatibility.
        probs = F.softmax(logits[mask], dim=-1).detach().cpu()
        y_true = labels[mask].detach().cpu().numpy()

        # Handle cases where the split doesn't have both classes (AUC is undefined)
        present = np.unique(y_true)
        if present.size < 2:
            return 0.5

        # Binary AUC uses the positive-class probability; multiclass uses
        # one-vs-rest restricted to the classes present in the split.
        if probs.size(1) == 2:
            return float(roc_auc_score(y_true, probs[:, 1].numpy()))
        y_bin = label_binarize(y_true, classes=np.arange(probs.size(1)))
        return float(
            roc_auc_score(y_bin[:, present], probs.numpy()[:, present], average="macro")
        )

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # noqa: ANN201
        """Setup Adam with a separate weight decay for the sheaf learners.

        Mirrors the reference optimizer: map-generator parameters get
        ``sheaf_decay`` (falling back to ``weight_decay``), everything else
        gets ``weight_decay``.
        """
        sheaf_decay = self.cfg.optim.weight_decay
        if getattr(self.cfg.optim, "sheaf_decay", None) is not None:
            sheaf_decay = self.cfg.optim.sheaf_decay
        sheaf_params, other_params = [], []
        for name, param in self.named_parameters():
            (sheaf_params if "map_generator" in name else other_params).append(param)
        if not sheaf_params:
            return torch.optim.Adam(
                other_params,
                lr=self.cfg.optim.lr,
                weight_decay=self.cfg.optim.weight_decay,
            )
        return torch.optim.Adam(
            [
                {"params": sheaf_params, "weight_decay": sheaf_decay},
                {"params": other_params, "weight_decay": self.cfg.optim.weight_decay},
            ],
            lr=self.cfg.optim.lr,
        )
