"""HSNN preset routing, config propagation and graph/hypergraph isolation."""

import copy
import sys

import pytest
import torch

from exp import cli, hsnn_run
from exp.config import ModelType
from exp.registries.models import model_registry
from exp.registries.presets import preset_registry


def hsnn_config():
    return copy.deepcopy(preset_registry.get("hsnn_cora"))


def test_cli_routes_preset_and_nested_overrides(monkeypatch, tmp_path):
    captured = {}

    def execute(args, cli_config=None):
        captured.update(vars(args))
        captured["config"] = cli_config
        return {"runs": [{"best_val_accuracy": 0.5, "test": None}]}

    monkeypatch.setattr(hsnn_run, "execute", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sheaf",
            "run",
            "--preset",
            "hsnn_cora",
            "--hypergraph.no-evaluate-test",
            "--hardware.cuda",
            "-1",
            "--model.variant",
            "low_rank",
            "--model.rank",
            "2",
            "--reg.input-dropout",
            "0.1",
            "--optim.epochs",
            "2",
            "--cv.folds",
            "1",
            "--hypergraph.output",
            str(tmp_path / "run"),
        ],
    )
    cli.main()
    assert captured["device"] == "cpu"
    assert captured["variant"] == "low_rank" and captured["rank"] == 2
    assert captured["input_dropout"] == 0.1
    assert captured["epochs"] == 2 and captured["runs"] == 1
    assert captured["patience"] == 200 and captured["seed"] == 0
    assert not captured["evaluate_test"]
    assert captured["data"].as_posix() == "data/cora-dshn/cora-dshn-q0.npz"
    assert captured["config"]["model"]["type"] == "hsnn"


def test_test_evaluation_is_default_and_returns_test_metrics(monkeypatch):
    cfg = hsnn_config()
    assert cfg.hypergraph.evaluate_test

    def execute(args, cli_config=None):
        assert args.evaluate_test
        return {"runs": [{"best_val_accuracy": 0.5, "test": {"accuracy": 0.4}}]}

    monkeypatch.setattr(hsnn_run, "execute", execute)
    assert hsnn_run.run(cfg) == [0.4]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("dataset", "name", "cora"),
        ("dataset", "directed", True),
        ("optim", "stop_strategy", "loss"),
        ("optim", "sheaf_decay", 0.1),
        ("model", "add_lp", True),
        ("cv", "min_acc", 0.2),
        ("wandb", "enabled", True),
    ],
)
def test_incompatible_protocol_overrides_fail_before_loading(section, key, value):
    cfg = hsnn_config()
    setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError):
        hsnn_run.run(cfg)


@pytest.mark.parametrize("variant", ["diagonal", "general", "orthogonal", "low_rank"])
def test_registered_model_uses_hyperedge_incidence(variant):
    cfg = hsnn_config()
    cfg.model.variant = variant
    cfg.model.stalk_dim = 2
    cfg.model.hidden_dim = 3
    cfg.model.rank = 1
    model = model_registry.build("hsnn", 3, 2, cfg.model, cfg.reg)
    # A hyperedge ID larger than the node count is valid; it is not a graph node.
    index = torch.tensor([[0, 1, 2, 1], [4, 4, 4, 0]])
    logits = model(torch.randn(3, 3), index, num_hyperedges=5)
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()
    logits.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_original_cora_preset_and_graph_runner_remain_separate(monkeypatch):
    from exp.run import run

    graph = preset_registry.get("cora")
    assert graph.model.type == ModelType.NSD
    assert graph.dataset.name == "cora"
    cfg = hsnn_config()
    cfg.model.type = ModelType.NSD
    with pytest.raises(ValueError, match="requires --model.type hsnn"):
        run(cfg)


def test_graph_lightning_rejects_hypergraph_model():
    from exp.data import DatasetInfo
    from exp.module import SheafLightningModule

    info = DatasetInfo(
        name="cora",
        num_features=3,
        num_classes=2,
        num_splits=10,
        metric="acc",
        split_type="npz_file",
    )
    with pytest.raises(ValueError, match="hypergraph runner"):
        SheafLightningModule(hsnn_config(), info)
