"""Directed hypergraph roles, registry and audited runner wiring."""

from pathlib import Path

import numpy as np
import pytest
import torch

from exp.config import ModelConfig, ModelType, RegConfig
from exp.dhsnn_run import load_roles, run
from exp.hsnn_run import load_data
from exp.registries.models import model_registry
from exp.registries.presets import preset_registry
from sheaf_mpnn import DHSNNModel


def test_real_roles_and_registry():
    path = Path("data/cora-dshn/cora-dshn-q0.npz")
    if path.exists():
        _, _, index, _ = load_data(path)
        roles = load_roles(path, index)
        assert roles.dtype == torch.long and (roles == -1).sum() == 1565
        assert (roles == 1).sum() == 5429
    model = model_registry.build(
        "dhsnn",
        3,
        2,
        ModelConfig(type=ModelType.DHSNN, variant="diagonal"),
        RegConfig(),
    )
    assert isinstance(model, DHSNNModel)
    cfg = preset_registry.get("dhsnn_cora")
    assert cfg.model.type == ModelType.DHSNN and cfg.dataset.directed


@pytest.mark.parametrize(
    "values", [np.array([-1.0, 1.0]), np.array([-1, 0]), np.array([1], dtype=np.int8)]
)
def test_reject_bad_roles(tmp_path, values):
    path = tmp_path / "data.npz"
    np.savez(path, incidence_roles=values)
    with pytest.raises(ValueError, match="incidence_roles"):
        load_roles(path, torch.tensor([[0, 1], [0, 0]]))


def test_directed_dispatch(monkeypatch, tmp_path):
    cfg = preset_registry.get("dhsnn_cora")
    cfg.hypergraph.output = str(tmp_path / "run")
    cfg.model.dhsnn_q = 0.17
    observed = []
    monkeypatch.setattr(
        "exp.dhsnn_run.execute",
        lambda args, cli_config: (
            observed.append(args)
            or {"runs": [{"best_val_accuracy": 0.7, "test": {"accuracy": 0.6}}]}
        ),
    )
    assert run(cfg) == [0.6]
    assert observed[0].evaluate_test
    assert observed[0].model == "dhsnn" and observed[0].q == 0.17


def test_graph_lightning_rejects_dhsnn():
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
        SheafLightningModule(preset_registry.get("dhsnn_cora"), info)


@pytest.mark.parametrize("model,preset", [("dhsnn", None), ("nsd", "dhsnn_cora")])
def test_graph_sweep_rejects_dhsnn(tmp_path, model, preset):
    from exp.sweeps.sweep import sweep

    path = tmp_path / "sweep.yaml"
    path.write_text(
        f"model: {model}\nsearch_space: {{}}\nconfig: {{n_trials: 1}}\n", "utf-8"
    )
    with pytest.raises(ValueError, match="unsupported|hypergraph preset"):
        sweep(path, preset=preset)
