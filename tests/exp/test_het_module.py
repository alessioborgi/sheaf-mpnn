# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data

from exp.config import Config, DatasetConfig, ModelConfig, ModelType, RegConfig
from exp.datasets.info import HetDatasetInfo
from exp.module import HetSheafLightningModule

NUM_NODES = 20
IN_CHANNELS = 16
OUT_CHANNELS = 3
NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 4


def _het_info(metric="acc", task="multiclass"):
    return HetDatasetInfo(
        name="test_het",
        num_features=IN_CHANNELS,
        num_classes=OUT_CHANNELS,
        num_splits=1,
        split_type="pyg_mask",
        metric=metric,
        task=task,
        num_node_types=NUM_NODE_TYPES,
        num_edge_types=NUM_EDGE_TYPES,
        node_type_names=tuple(f"type_{i}" for i in range(NUM_NODE_TYPES)),
        edge_type_names=tuple(f"etype_{i}" for i in range(NUM_EDGE_TYPES)),
    )


def _cfg(model_type=ModelType.HET_NSD, variant="general"):
    return Config(
        dataset=DatasetConfig(name="imdb"),
        model=ModelConfig(
            type=model_type,
            variant=variant,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            use_features=True,
            use_node_types=True,
            use_edge_types=True,
        ),
        reg=RegConfig(),
    )


def _batch(with_types=True):
    """Create a minimal homogeneous Data batch with optional type tensors."""
    N, E = NUM_NODES, 30
    edge_index = torch.randint(0, N, (2, E))
    data = Data(
        x=torch.randn(N, IN_CHANNELS),
        edge_index=edge_index,
        y=torch.randint(0, OUT_CHANNELS, (N,)),
        train_mask=torch.zeros(N, dtype=torch.bool).scatter_(0, torch.arange(10), True),
        val_mask=torch.zeros(N, dtype=torch.bool).scatter_(
            0, torch.arange(10, 15), True
        ),
        test_mask=torch.zeros(N, dtype=torch.bool).scatter_(
            0, torch.arange(15, N), True
        ),
    )
    if with_types:
        data.node_type = torch.randint(0, NUM_NODE_TYPES, (N,))
        data.edge_type = torch.randint(0, NUM_EDGE_TYPES, (E,))
    return data


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type,variant",
    [
        (ModelType.HET_NSD, "general"),
        (ModelType.HET_NSD, "diagonal"),
        (ModelType.HET_NSD, "orthogonal"),
        (ModelType.HET_NSP, "general"),
        (ModelType.HET_SHEAF_ATTN, "sheafan"),
        (ModelType.HET_SHEAF_ATTN, "res_sheafan"),
        (ModelType.HET_ID_SHEAF, "general"),
        (ModelType.HET_JOINT_DIFF, "general"),
        (ModelType.HET_ROT_INV, "general"),
        (ModelType.HET_DIRECTED_NSD, "general"),
        (ModelType.HET_DIRECTED_NSD, "orthogonal"),
        (ModelType.HET_BSNN, "diagonal"),
        (ModelType.HET_BSNN, "orthogonal"),
    ],
)
def test_construction(model_type, variant):
    cfg = _cfg(model_type, variant)
    info = _het_info()
    module = HetSheafLightningModule(cfg, info)
    assert module.model is not None


def test_construction_het_csnn():
    """HET_CSNN needs use_edge_types off: the default SAGE learner backbone
    consumes no edge features, and the builder rejects the silent drop.
    """
    cfg = _cfg(ModelType.HET_CSNN, "conformal")
    cfg.model.__dict__["use_edge_types"] = False
    module = HetSheafLightningModule(cfg, _het_info())
    assert module.model is not None


def test_het_csnn_edge_types_rejected_on_default_backbone():
    cfg = _cfg(ModelType.HET_CSNN, "conformal")
    with pytest.raises(ValueError, match="use_edge_types"):
        HetSheafLightningModule(cfg, _het_info())


def test_het_bunn_training_step_with_types():
    """HET_BUNN reads edge types through its SumGNN phi, so the config switches
    phi to sum_gnn; the default MLP phi rejects use_edge_types.
    """
    cfg = _cfg(ModelType.HET_BUNN, "orthogonal")
    cfg.model.__dict__["bunn_phi"] = "sum_gnn"
    module = HetSheafLightningModule(cfg, _het_info())
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    loss = module.training_step(_batch(with_types=True), 0)
    assert torch.isfinite(loss)


def test_het_bunn_edge_types_rejected_with_mlp_phi():
    cfg = _cfg(ModelType.HET_BUNN, "orthogonal")
    with pytest.raises(ValueError, match="use_edge_types"):
        HetSheafLightningModule(cfg, _het_info())


def test_unknown_model_type_raises():
    cfg = _cfg()
    cfg.model.__dict__["type"] = "nonexistent_type"
    with pytest.raises(ValueError, match="Unknown model type"):
        HetSheafLightningModule(cfg, _het_info())


# ---------------------------------------------------------------------------
# Forward / step
# ---------------------------------------------------------------------------


def test_training_step_with_types():
    cfg = _cfg()
    module = HetSheafLightningModule(cfg, _het_info())
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    batch = _batch(with_types=True)
    loss = module.training_step(batch, 0)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_training_step_without_types():
    """Falls back gracefully when node_type/edge_type are absent."""
    cfg = _cfg(ModelType.HET_NSD, "general")
    cfg.model.__dict__["use_node_types"] = False
    cfg.model.__dict__["use_edge_types"] = False
    cfg.model.__dict__["use_features"] = True
    module = HetSheafLightningModule(cfg, _het_info())
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    batch = _batch(with_types=False)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_validation_step_logs_metric():
    cfg = _cfg()
    module = HetSheafLightningModule(cfg, _het_info())
    logged: dict = {}
    module.log = lambda k, v, **kw: logged.update(  # ty: ignore[invalid-assignment]
        {k: v}
    )
    batch = _batch(with_types=True)
    module.validation_step(batch, 0)
    assert "val_loss" in logged
    assert "val_acc" in logged


def test_test_step_logs_metric():
    cfg = _cfg()
    module = HetSheafLightningModule(cfg, _het_info())
    logged: dict = {}
    module.log = lambda k, v, **kw: logged.update(  # ty: ignore[invalid-assignment]
        {k: v}
    )
    batch = _batch(with_types=True)
    module.test_step(batch, 0)
    assert "test_loss" in logged
    assert "test_acc" in logged


def test_nan_logits_handled():
    """NaN logits trigger inf-loss fallback without crashing."""
    cfg = _cfg()
    module = HetSheafLightningModule(cfg, _het_info())
    logged: dict = {}
    module.log = lambda k, v, **kw: logged.update(  # ty: ignore[invalid-assignment]
        {k: float(v)}
    )

    # Monkeypatch model forward to return NaN (bypass nn.Module's __setattr__).
    nan_out = torch.full((NUM_NODES, OUT_CHANNELS), float("nan"))
    module.model.forward = MagicMock(return_value=nan_out)
    batch = _batch(with_types=True)
    module.validation_step(batch, 0)
    assert logged["val_loss"] == float("inf")
    assert logged["val_acc"] == 0.0


@pytest.mark.parametrize(
    "model_type,variant",
    [
        (ModelType.HET_ID_SHEAF, "general"),
        (ModelType.HET_JOINT_DIFF, "general"),
        (ModelType.HET_ROT_INV, "general"),
        (ModelType.HET_DIRECTED_NSD, "general"),
    ],
)
def test_training_step_new_het_models(model_type, variant):
    cfg = _cfg(model_type, variant)
    module = HetSheafLightningModule(cfg, _het_info())
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    batch = _batch(with_types=True)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_het_bsnn_elbo_training_step():
    """HET_BSNN trains through the het Bayes module: CE + annealed KL."""
    from exp.module import HetBayesSheafLightningModule

    cfg = _cfg(ModelType.HET_BSNN, "diagonal")
    module = HetBayesSheafLightningModule(cfg, _het_info())
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    batch = _batch(with_types=True)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert torch.isfinite(module.model.kl_loss())


def test_het_bsnn_ensembled_eval_forwards_types():
    """Evaluation ensembling passes the type tensors to predict_proba."""
    from exp.module import HetBayesSheafLightningModule

    cfg = _cfg(ModelType.HET_BSNN, "diagonal")
    module = HetBayesSheafLightningModule(cfg, _het_info())
    module.eval()
    captured: dict = {}
    original = module.model.predict_proba

    def capturing(x, edge_index, node_types=None, edge_types=None, num_samples=None):
        captured["node_types"] = node_types
        captured["edge_types"] = edge_types
        return original(x, edge_index, node_types, edge_types, num_samples)

    module.model.predict_proba = capturing
    batch = _batch(with_types=True)
    logits = module._forward(batch)
    assert logits.shape == (NUM_NODES, OUT_CHANNELS)
    assert captured["node_types"] is not None
    assert captured["edge_types"] is not None


# ---------------------------------------------------------------------------
# Type tensors passed to model
# ---------------------------------------------------------------------------


def test_type_tensors_forwarded_to_model():
    """HetSheafLightningModule passes node_type and edge_type to model.forward."""
    cfg = _cfg()
    module = HetSheafLightningModule(cfg, _het_info())
    captured: dict = {}

    original_forward = module.model.forward

    def capturing_forward(x, edge_index, node_types=None, edge_types=None):
        captured["node_types"] = node_types
        captured["edge_types"] = edge_types
        return original_forward(x, edge_index, node_types, edge_types)

    module.model.forward = capturing_forward
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    batch = _batch(with_types=True)
    module.training_step(batch, 0)

    assert captured["node_types"] is not None
    assert captured["edge_types"] is not None


def test_missing_type_tensors_pass_none():
    """When data has no node_type, None is passed to the model."""
    cfg = _cfg()
    cfg.model.__dict__["use_node_types"] = False
    cfg.model.__dict__["use_edge_types"] = False
    module = HetSheafLightningModule(cfg, _het_info())
    captured: dict = {}

    original_forward = module.model.forward

    def capturing_forward(x, edge_index, node_types=None, edge_types=None):
        captured["node_types"] = node_types
        captured["edge_types"] = edge_types
        return original_forward(x, edge_index, node_types, edge_types)

    module.model.forward = capturing_forward
    module.log = MagicMock()  # ty: ignore[invalid-assignment]
    batch = _batch(with_types=False)
    module.training_step(batch, 0)

    assert captured["node_types"] is None
    assert captured["edge_types"] is None


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_configure_optimizers():
    cfg = _cfg()
    module = HetSheafLightningModule(cfg, _het_info())
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == pytest.approx(cfg.optim.lr)
