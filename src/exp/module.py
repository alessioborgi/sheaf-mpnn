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

from exp.config import Config, ModelType
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
        kw: dict = {"on_step": False, "on_epoch": True, "batch_size": 1}
        if getattr(self.info, "task", "multiclass") == "multilabel":
            probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
            y_true = labels[mask].detach().cpu().numpy().astype(int)
            y_pred = (probs > 0.5).astype(int)
            for avg in ("macro", "micro"):
                self.log(
                    f"{prefix}_f1_{avg}",
                    float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
                    **kw,
                )
            self.log(
                f"{prefix}_precision_macro",
                float(
                    precision_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                **kw,
            )
            self.log(
                f"{prefix}_recall_macro",
                float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                **kw,
            )
            # AP (like AUC) is undefined for targets with a single class in
            # the split; average over the defined ones only.
            present = (y_true.min(axis=0) < y_true.max(axis=0)).nonzero()[0]
            if present.size:
                self.log(
                    f"{prefix}_pr_auc",
                    float(
                        average_precision_score(
                            y_true[:, present], probs[:, present], average="macro"
                        )
                    ),
                    **kw,
                )
            return

        probs = F.softmax(logits[mask], dim=-1).detach().cpu().numpy()
        y_true = labels[mask].detach().cpu().numpy()
        y_pred = probs.argmax(axis=-1)
        n_classes = probs.shape[1]

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
        if getattr(self.info, "task", "multiclass") == "multilabel":
            # Mean AUC over the binary targets whose split contains both
            # classes (the OGB ogbn-proteins protocol).
            probs_ml = torch.sigmoid(logits[mask]).detach().cpu().numpy()
            y_ml = labels[mask].detach().cpu().numpy().astype(int)
            defined = (y_ml.min(axis=0) < y_ml.max(axis=0)).nonzero()[0]
            if defined.size == 0:
                return 0.5
            return float(
                roc_auc_score(y_ml[:, defined], probs_ml[:, defined], average="macro")
            )
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

        NLSD additionally trains its nonlinearity threshold and its per-stalk
        diffusion gate at their own learning rates (reference ``threshold_lr``
        / ``alpha_lr``).  Both are opt-in: leaving them ``None`` keeps every
        parameter on ``optim.lr``, so nothing changes for the other families.
        """
        optim = self.cfg.optim
        sheaf_decay = optim.weight_decay
        if getattr(optim, "sheaf_decay", None) is not None:
            sheaf_decay = optim.sheaf_decay

        threshold_lr = getattr(optim, "threshold_lr", None)
        alpha_lr = getattr(optim, "alpha_lr", None)

        sheaf_params, threshold_params, alpha_params, other_params = [], [], [], []
        for name, param in self.named_parameters():
            # map_learner: BuNN's per-node bundle learner phi, the flat-bundle
            # analogue of the sheaf families' per-edge map_generator.
            if "map_generator" in name or "map_learner" in name:
                sheaf_params.append(param)
            elif threshold_lr is not None and "nonlinearity.threshold" in name:
                threshold_params.append(param)
            elif alpha_lr is not None and "alpha_gate" in name:
                alpha_params.append(param)
            else:
                other_params.append(param)

        groups: list[dict] = []
        if sheaf_params:
            groups.append({"params": sheaf_params, "weight_decay": sheaf_decay})
        if threshold_params:
            groups.append(
                {
                    "params": threshold_params,
                    "weight_decay": optim.weight_decay,
                    "lr": threshold_lr,
                }
            )
        if alpha_params:
            groups.append(
                {
                    "params": alpha_params,
                    "weight_decay": optim.weight_decay,
                    "lr": alpha_lr,
                }
            )
        if not groups:
            return torch.optim.Adam(
                other_params, lr=optim.lr, weight_decay=optim.weight_decay
            )
        groups.append({"params": other_params, "weight_decay": optim.weight_decay})
        return torch.optim.Adam(groups, lr=optim.lr)


class BayesSheafLightningModule(SheafLightningModule):
    """Extends SheafLightningModule for Bayesian sheaf models (BSNN).

    Training maximizes the ELBO: cross-entropy plus a KL divergence
    regularizer whose weight is cyclically annealed with a sigmoid ramp
    (Fu et al., 2019), matching the reference BSNN training loop
    ``beta = sigmoid((epoch % 40) / 2 - 10)``.

    Evaluation ensembles ``cfg.model.ensemble_size`` stochastic forward
    passes: softmax probabilities are averaged and their log is returned as
    the logits (log-probabilities are fixed points of log-softmax, so every
    downstream loss/metric computation stays valid). The per-pass ensemble is
    kept around to log the uncertainty-quantification metrics of Gillespie
    et al. (2025, Section 2.5).
    """

    # Floor for ensemble-averaged probabilities before the log: keeps the
    # returned logits finite when a class probability underflows to zero.
    _PROB_EPS = 1e-12

    def __init__(self, cfg: Config, info: DatasetInfo) -> None:
        super().__init__(cfg, info)
        self._last_ensemble: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # ELBO training
    # ------------------------------------------------------------------

    def kl_beta(self) -> float:
        """Current weight of the KL term in the ELBO.

        The cyclical schedule ramps 0 -> ``kl_weight`` with a sigmoid over
        each ``kl_cycle``-epoch window; the reference schedule is recovered
        with ``kl_weight=1`` and ``kl_cycle=40``.
        """
        m = self.cfg.model
        if not m.use_kl:
            return 0.0
        if m.kl_anneal == "constant":
            return m.kl_weight
        cycle = max(int(m.kl_cycle), 1)
        frac = (self.current_epoch % cycle) / cycle
        return m.kl_weight * float(torch.sigmoid(torch.tensor(20.0 * frac - 10.0)))

    def training_step(self, batch, batch_idx):
        """ELBO training step: cross-entropy + annealed KL divergence."""
        data = batch
        logits = self._forward(data)

        if getattr(self.info, "task", "multiclass") == "multilabel":
            ce = F.binary_cross_entropy_with_logits(
                logits[data.train_mask], data.y[data.train_mask].float()
            )
        else:
            ce = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        kl = self.model.kl_loss()
        beta = self.kl_beta()
        loss = ce + beta * kl if self.cfg.model.use_kl else ce

        kw: dict = {"on_step": False, "on_epoch": True, "batch_size": 1}
        self.log("train_loss", loss, prog_bar=True, **kw)
        self.log("train_ce", ce, **kw)
        self.log("train_kl", kl, **kw)
        self.log("kl_beta", beta, **kw)
        return loss

    # ------------------------------------------------------------------
    # Ensembled evaluation
    # ------------------------------------------------------------------

    def _forward(self, data) -> torch.Tensor:
        """Single stochastic pass in training; prob-averaged ensemble in eval.

        Multilabel tasks skip the softmax ensembling (probabilities are
        per-class sigmoids there) and fall back to a single pass.
        """
        self._last_ensemble = None
        ensemble = int(self.cfg.model.ensemble_size)
        multilabel = getattr(self.info, "task", "multiclass") == "multilabel"
        if self.training or ensemble <= 1 or multilabel:
            return self.model(data.x, data.edge_index)

        probs = self.model.predict_proba(data.x, data.edge_index, ensemble)
        self._last_ensemble = probs
        return (probs.mean(dim=0) + self._PROB_EPS).log()

    def _log_extra_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        prefix: str,
    ) -> None:
        super()._log_extra_metrics(logits, labels, mask, prefix)
        if self._last_ensemble is None:
            return
        from sheaf_mpnn.bsnn import uncertainty_summary

        summary = uncertainty_summary(self._last_ensemble[:, mask], labels[mask])
        kw: dict = {"on_step": False, "on_epoch": True, "batch_size": 1}
        self.log(f"{prefix}_pred_entropy", summary["entropy"], **kw)
        self.log(f"{prefix}_epistemic_var", summary["epistemic_variance"], **kw)
        self.log(f"{prefix}_mutual_info", summary["mutual_information"], **kw)
        self.log(f"{prefix}_ece", summary["ece"], **kw)

    # ------------------------------------------------------------------
    # Optimizer configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # noqa: ANN201
        """Adam with ``sheaf_decay`` on the variational sheaf learner.

        The BSNN learner heads are named ``sheaf_learner`` rather than
        ``map_generator``; group them under the sheaf weight decay to mirror
        the reference optimizer's grouped parameters.
        """
        sheaf_decay = self.cfg.optim.weight_decay
        if getattr(self.cfg.optim, "sheaf_decay", None) is not None:
            sheaf_decay = self.cfg.optim.sheaf_decay
        sheaf_params, other_params = [], []
        for name, param in self.named_parameters():
            is_sheaf = "sheaf_learner" in name or "map_generator" in name
            (sheaf_params if is_sheaf else other_params).append(param)
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


def select_module_cls(cfg: Config) -> type[SheafLightningModule]:
    """Lightning module class for the configured model type.

    Bayesian sheaf models need the ELBO loss and ensembled evaluation of
    ``BayesSheafLightningModule``; every other model keeps the standard
    module.
    """
    if cfg.model.type == ModelType.BSNN:
        return BayesSheafLightningModule
    return SheafLightningModule


class HetSheafLightningModule(SheafLightningModule):
    """Extends SheafLightningModule for heterogeneous graphs.

    Passes ``node_type`` and ``edge_type`` tensors (produced by PyG's
    ``to_homogeneous()``) to the model at each forward call.  Falls back
    gracefully when these attributes are absent (e.g. homogeneous datasets).
    """

    def __init__(self, cfg: Config, info: DatasetInfo) -> None:
        from exp.datasets.info import HetDatasetInfo

        if isinstance(info, HetDatasetInfo) and "HET_" in str(cfg.model.type).upper():
            if cfg.model.num_node_types == 0:
                cfg.model.num_node_types = info.num_node_types
            if cfg.model.num_edge_types == 0:
                cfg.model.num_edge_types = info.num_edge_types
        super().__init__(cfg, info)

    def _forward(self, data) -> torch.Tensor:
        model_type = str(self.cfg.model.type).upper()
        if "HET_" not in model_type:
            return super()._forward(data)
        node_types = getattr(data, "node_type", None)
        edge_types = getattr(data, "edge_type", None)
        return self.model(data.x, data.edge_index, node_types, edge_types)


class HetBayesSheafLightningModule(BayesSheafLightningModule, HetSheafLightningModule):
    """Heterogeneous Bayesian sheaf module (HET_BSNN).

    Combines the ELBO training and ensembled evaluation of
    ``BayesSheafLightningModule`` with the type-tensor threading of
    ``HetSheafLightningModule``: the MRO runs the het ``__init__`` (which
    back-fills ``num_node_types`` / ``num_edge_types`` from the dataset), and
    ``_forward`` passes the types through both the stochastic pass and the
    evaluation ensemble.
    """

    def _forward(self, data) -> torch.Tensor:
        self._last_ensemble = None
        node_types = getattr(data, "node_type", None)
        edge_types = getattr(data, "edge_type", None)
        ensemble = int(self.cfg.model.ensemble_size)
        multilabel = getattr(self.info, "task", "multiclass") == "multilabel"
        if self.training or ensemble <= 1 or multilabel:
            return self.model(data.x, data.edge_index, node_types, edge_types)

        probs = self.model.predict_proba(
            data.x, data.edge_index, node_types, edge_types, ensemble
        )
        self._last_ensemble = probs
        return (probs.mean(dim=0) + self._PROB_EPS).log()
