# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Lightning module for sheaf-based link prediction.

The sheaf backbone encodes all nodes in a homogeneous graph to an embedding
space of dimension ``hidden_dim * stalk_dim`` (the model's internal
``context_dim``).  Edge existence is scored with the inner product of the
two endpoint embeddings; the loss is binary cross-entropy.

Evaluation metrics: AUROC and AUPR (average precision).

The DataModule must deliver plain homogeneous batches (``Data`` objects) with:
  * ``batch.x``               - node features ``[N, in_channels]``
  * ``batch.edge_index``      - message-passing edges ``[2, E_mp]``
  * ``batch.edge_label_index``- supervised edge pairs (pos + neg) ``[2, E_sup]``
  * ``batch.edge_label``      - binary labels ``[E_sup]`` (float32, 0.0 / 1.0)

Use :class:`~exp.datasets.link_pred.LinkPredBase` with ``is_homogeneous=True``
to obtain this format from heterogeneous source graphs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from sklearn.metrics import average_precision_score, roc_auc_score

from exp.config import Config
from exp.datasets.info import LPDatasetInfo
from exp.registries.models import model_registry


class LPLightningModule(LightningModule):
    """Sheaf model encoder + inner-product scorer for link prediction.

    Args:
        cfg: Full experiment configuration.
        info: Link-prediction dataset metadata (must have ``num_features``).
    """

    def __init__(self, cfg: Config, info: LPDatasetInfo) -> None:
        super().__init__()
        self.cfg = cfg
        self.info = info
        # Use context_dim (hidden_dim * stalk_dim) as the embedding dimension.
        # This reuses the model's internal representation size and avoids adding
        # a new config field.
        emb_dim = cfg.model.hidden_dim * cfg.model.stalk_dim
        try:
            self.model: Any = model_registry.build(
                str(cfg.model.type),
                info.num_features,
                emb_dim,
                cfg.model,
                cfg.reg,
            )
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unknown model type: {cfg.model.type!r}") from exc

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Inner-product link score for each candidate edge pair.

        Args:
            z: Node embeddings ``[N, emb_dim]``.
            edge_label_index: Supervised edge pairs ``[2, E_sup]``.

        Returns:
            Raw logits ``[E_sup]``.
        """
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Encode nodes, score supervised edges, compute BCE loss."""
        z = self.model(batch.x, batch.edge_index)
        logits = self._score(z, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluate on validation edges."""
        self._eval_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluate on test edges."""
        self._eval_step(batch, "test")

    def _eval_step(self, batch: Any, prefix: str) -> None:
        z = self.model(batch.x, batch.edge_index)
        logits = self._score(z, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)

        prog = prefix == "val"
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.edge_label.detach().cpu().numpy()

        if np.unique(labels).size < 2:
            auroc, aupr = 0.5, 0.0
        else:
            auroc = float(roc_auc_score(labels, probs))
            aupr = float(average_precision_score(labels, probs))

        self.log(
            f"{prefix}_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )
        self.log(
            f"{prefix}_aupr",
            aupr,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam with lr and weight_decay from config."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )


class HetLPLightningModule(LPLightningModule):
    """Extends LPLightningModule for heterogeneous link prediction.

    Passes ``node_type`` and ``edge_type`` tensors to the model at each forward call.
    """

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Encode nodes, score supervised edges, compute BCE loss."""
        node_types = getattr(batch, "node_type", None)
        edge_types = getattr(batch, "edge_type", None)
        z = self.model(batch.x, batch.edge_index, node_types, edge_types)
        logits = self._score(z, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )
        return loss

    def _eval_step(self, batch: Any, prefix: str) -> None:
        node_types = getattr(batch, "node_type", None)
        edge_types = getattr(batch, "edge_type", None)
        z = self.model(batch.x, batch.edge_index, node_types, edge_types)
        logits = self._score(z, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)

        prog = prefix == "val"
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.edge_label.detach().cpu().numpy()

        if np.unique(labels).size < 2:
            auroc, aupr = 0.5, 0.0
        else:
            auroc = float(roc_auc_score(labels, probs))
            aupr = float(average_precision_score(labels, probs))

        self.log(
            f"{prefix}_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )
        self.log(
            f"{prefix}_aupr",
            aupr,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=1,
        )
