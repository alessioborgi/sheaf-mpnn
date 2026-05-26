# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

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
from sklearn.metrics import roc_auc_score

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

    def training_step(self, batch, batch_idx):
        """Standard training step: forward pass + cross-entropy loss."""
        data = batch
        # Forward pass through the sheaf model
        logits = self.model(data.x, data.edge_index)

        # Calculate loss only on the training subset
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
        logits = self.model(data.x, data.edge_index)

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
        # Case 1: Multi-class Accuracy
        if self.info.metric == "acc":
            pred = logits[mask].argmax(dim=-1)
            return float(pred.eq(labels[mask]).sum().item()) / int(mask.sum().item())

        # Case 2: ROC-AUC (used for heterophilous/binary tasks)
        # AUC is computed on CPU probabilities for sklearn compatibility.
        probs = F.softmax(logits[mask], dim=-1).detach().cpu()
        y_true = labels[mask].detach().cpu().numpy()

        # Handle cases where the split doesn't have both classes (AUC is undefined)
        if np.unique(y_true).size < 2:
            return 0.5

        # Binary AUC uses the positive-class probability; multiclass uses one-vs-rest.
        if probs.size(1) == 2:
            return float(roc_auc_score(y_true, probs[:, 1].numpy()))
        return float(
            roc_auc_score(
                y_true,
                probs.numpy(),
                multi_class="ovr",
                labels=np.arange(probs.size(1)),
            )
        )

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # noqa: ANN201
        """Setup Adam optimizer with parameters from the config."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
