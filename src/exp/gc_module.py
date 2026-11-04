# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Lightning module for sheaf-based graph classification.

The sheaf model acts as a node-level encoder; its output logits are pooled to
graph level before the cross-entropy loss is applied, using one of five
strategies selected by ``GCPooling``:

* ``MEAN``       - :func:`~torch_geometric.nn.global_mean_pool` (default)
* ``ADD``        - :func:`~torch_geometric.nn.global_add_pool`
* ``MAX``        - :func:`~torch_geometric.nn.global_max_pool`
* ``SAG``        - :class:`~torch_geometric.nn.SAGPooling` (self-attention graph
                   pooling, Lee et al., 2019) followed by mean-pooling of the
                   retained nodes.
* ``SHEAF_POOL`` - :class:`~sheaf_mpnn.sheaf_pool.SheafPool` gauge-invariant
                   readout operating directly on stalk embeddings
                   (HetSheaf paper, arXiv:2409.08036v3).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch_geometric.nn import (
    SAGPooling,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from exp.config import Config, GCPooling
from exp.datasets.info import GCDatasetInfo
from exp.registries.models import model_registry
from sheaf_mpnn.sheaf_pool import SheafPool


class GCLightningModule(LightningModule):
    """Sheaf model + pooling head for graph classification.

    The backbone acts as a pure stalk encoder: its ``nn.Linear`` decoder is
    replaced with ``nn.Identity`` so ``forward()`` returns raw node stalk
    features ``[N, d*C]``.  A graph-level classifier (``gc_classifier``) owned
    by this module then maps the pooled representation to class logits.

    All five pooling strategies use the same ``forward()`` entry point:

    * ``MEAN / ADD / MAX``  — standard PyG global-pool on ``[N, d*C]`` features
    * ``SAG``               — :class:`~torch_geometric.nn.SAGPooling` on ``[N, d*C]``
    * ``SHEAF_POOL``        — :class:`~sheaf_mpnn.sheaf_pool.SheafPool` on ``[N, d*C]``
                              (SheafPool accepts flat input and reshapes internally)

    Args:
        cfg: Full experiment configuration.
        info: Graph-classification dataset metadata.
    """

    def __init__(
        self,
        cfg: Config,
        info: GCDatasetInfo,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.info = info
        self.pooling = cfg.model.pooling
        try:
            self.model: Any = model_registry.build(
                str(cfg.model.type),
                info.num_features,
                info.num_classes,
                cfg.model,
                cfg.reg,
            )
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unknown model type: {cfg.model.type!r}") from exc

        # Replace the model's decoder with Identity so forward() returns [N, d*C].
        self.model.decoder = nn.Identity()

        context_dim = cfg.model.stalk_dim * cfg.model.hidden_dim

        # SAGPooling operates on the raw [N, d*C] stalk features.
        if self.pooling == GCPooling.SAG:
            self.sag_pool = SAGPooling(context_dim, ratio=0.5)

        # SheafPool accepts flat [N, d*C] input and produces [B, hidden_dim].
        if self.pooling == GCPooling.SHEAF_POOL:
            self.sheaf_pool = SheafPool(
                stalk_dim=cfg.model.stalk_dim,
                channel_dim=cfg.model.hidden_dim,
            )
            self.gc_classifier: nn.Linear = nn.Linear(
                cfg.model.hidden_dim, info.num_classes
            )
        else:
            self.gc_classifier = nn.Linear(context_dim, info.num_classes)

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------

    def _graph_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute graph-level class logits for a batch of graphs.

        The backbone ``forward()`` returns raw stalk features ``[N, d*C]``
        (Identity decoder).  These are pooled to ``[B, pool_dim]`` and then
        passed through ``gc_classifier`` to produce ``[B, num_classes]``.

        Extra keyword arguments (e.g. ``node_types``, ``edge_types`` for het
        models) are forwarded to the backbone unchanged.
        """
        node_features = self.model(x, edge_index, **kwargs)  # [N, d*C]
        if self.pooling == GCPooling.SHEAF_POOL:
            graph_emb = self.sheaf_pool(node_features, batch)  # [B, hidden_dim]
        else:
            graph_emb = self._pool(node_features, edge_index, batch)  # [B, d*C]
        return self.gc_classifier(graph_emb)  # [B, num_classes]

    def _pool(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Pool node stalk features (N, d*C) to graph-level embeddings (B, d*C)."""
        if self.pooling == GCPooling.ADD:
            return global_add_pool(node_features, batch)
        if self.pooling == GCPooling.MAX:
            return global_max_pool(node_features, batch)
        if self.pooling == GCPooling.SAG:
            # SAGPooling selects the ratio most informative nodes via self-attention,
            # then mean-pools the retained subset (Lee et al., 2019).
            x_pooled, _, _, batch_p, _, _ = self.sag_pool(
                node_features, edge_index, batch=batch
            )
            return global_mean_pool(x_pooled, batch_p)
        # Default: MEAN
        return global_mean_pool(node_features, batch)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Encode -> pool -> cross-entropy on graph label."""
        graph_logits = self._graph_logits(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(graph_logits, batch.y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(batch.y.size(0)),
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluate on validation graphs."""
        self._eval_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Evaluate on test graphs."""
        self._eval_step(batch, "test")

    def _eval_step(self, batch: Any, prefix: str) -> None:
        graph_logits = self._graph_logits(batch.x, batch.edge_index, batch.batch)

        if not torch.isfinite(graph_logits).all():  # pragma: no cover
            inf = torch.tensor(float("inf"))
            self.log(
                f"{prefix}_loss",
                inf,
                on_step=False,
                on_epoch=True,
                prog_bar=(prefix == "val"),
                batch_size=int(batch.y.size(0)),
            )
            self.log(
                f"{prefix}_acc",
                torch.tensor(0.0),
                on_step=False,
                on_epoch=True,
                prog_bar=(prefix == "val"),
                batch_size=int(batch.y.size(0)),
            )
            return

        loss = F.cross_entropy(graph_logits, batch.y)
        pred = graph_logits.argmax(dim=-1)
        acc = float(pred.eq(batch.y).sum().item()) / int(batch.y.size(0))

        prog = prefix == "val"
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=int(batch.y.size(0)),
        )
        self.log(
            f"{prefix}_acc",
            torch.tensor(acc),
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=int(batch.y.size(0)),
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


class HetGCLightningModule(GCLightningModule):
    """Extends GCLightningModule for heterogeneous graph classification.

    Passes ``node_type`` and ``edge_type`` tensors to the model at each forward call.
    ``_graph_logits`` is inherited from the base class; the het-specific kwargs
    (``node_types``, ``edge_types``) are forwarded via ``**kwargs``.
    """

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Node logits -> pool -> cross-entropy on graph label."""
        node_types = getattr(batch, "node_type", None)
        edge_types = getattr(batch, "edge_type", None)
        graph_logits = self._graph_logits(
            batch.x,
            batch.edge_index,
            batch.batch,
            node_types=node_types,
            edge_types=edge_types,
        )
        loss = F.cross_entropy(graph_logits, batch.y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(batch.y.size(0)),
        )
        return loss

    def _eval_step(self, batch: Any, prefix: str) -> None:
        node_types = getattr(batch, "node_type", None)
        edge_types = getattr(batch, "edge_type", None)
        graph_logits = self._graph_logits(
            batch.x,
            batch.edge_index,
            batch.batch,
            node_types=node_types,
            edge_types=edge_types,
        )

        if not torch.isfinite(graph_logits).all():  # pragma: no cover
            inf = torch.tensor(float("inf"))
            self.log(
                f"{prefix}_loss",
                inf,
                on_step=False,
                on_epoch=True,
                prog_bar=(prefix == "val"),
                batch_size=int(batch.y.size(0)),
            )
            self.log(
                f"{prefix}_acc",
                torch.tensor(0.0),
                on_step=False,
                on_epoch=True,
                prog_bar=(prefix == "val"),
                batch_size=int(batch.y.size(0)),
            )
            return

        loss = F.cross_entropy(graph_logits, batch.y)
        pred = graph_logits.argmax(dim=-1)
        acc = float(pred.eq(batch.y).sum().item()) / int(batch.y.size(0))

        prog = prefix == "val"
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=int(batch.y.size(0)),
        )
        self.log(
            f"{prefix}_acc",
            torch.tensor(acc),
            on_step=False,
            on_epoch=True,
            prog_bar=prog,
            batch_size=int(batch.y.size(0)),
        )
