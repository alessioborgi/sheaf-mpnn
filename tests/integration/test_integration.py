import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from lightning import Trainer

from exp.config import (
    Config,
    DatasetConfig,
    ModelConfig,
    ModelType,
    OptimConfig,
    RegConfig,
)
from exp.data import SheafDataModule
from exp.gen_splits import SplitsConfig, splits
from exp.module import SheafLightningModule
from exp.sweeps.sweep import sweep

DATA_DIR = "exp/data"
MAX_EPOCHS = 3


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _minimal_cfg(
    dataset_name="cora",
    model_type=ModelType.NSD,
    variant="general",
    normalize_output=True,
):
    return Config(
        dataset=DatasetConfig(name=dataset_name, root=DATA_DIR),
        model=ModelConfig(
            type=model_type,
            variant=variant,
            stalk_dim=2,
            hidden_dim=4,
            num_layers=1,
            normalize_output=normalize_output,
        ),
        reg=RegConfig(input_dropout=0.0, dropout=0.0),
        optim=OptimConfig(
            lr=0.01,
            epochs=MAX_EPOCHS,
            early_stopping=MAX_EPOCHS + 1,
            batch_size=1,
        ),
    )


def _trainer():
    return Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )


def _check_weight_updates(module, datamodule):
    """Verify that model parameters are actually updated during training."""
    # Deep copy parameters before training
    before_params = [p.clone().detach() for p in module.parameters()]

    t = _trainer()
    t.fit(module, datamodule)

    # Check that at least one parameter has changed
    after_params = [p.detach() for p in module.parameters()]
    changed = False
    for b, a in zip(before_params, after_params, strict=False):
        if not torch.equal(b, a):
            changed = True
            break

    assert changed, "No parameters were updated during training! Check gradient flow."
    return t


NSD_VARIANTS = [
    (ModelType.NSD, "diagonal"),
    (ModelType.NSD, "general"),
    (ModelType.NSD, "orthogonal"),
    (ModelType.NSD, "low_rank"),
    (ModelType.NSD, "general_attention"),
    (ModelType.NSD, "orthogonal_attention"),
]


@pytest.mark.integration
@pytest.mark.parametrize("model_type,variant", NSD_VARIANTS)
def test_nc_homogeneous_exhaustive(model_type, variant):
    """Exhaustive test for Node Classification with all NSD variants."""
    dm = SheafDataModule("cora", root=DATA_DIR)
    dm.setup()
    cfg = _minimal_cfg(model_type=model_type, variant=variant)
    module = SheafLightningModule(cfg, dm.info)

    t = _check_weight_updates(module, dm)
    [res] = t.test(module, dm, verbose=False)
    assert res["test_acc"] > 1.0 / dm.info.num_classes


@pytest.mark.integration
def test_splits_generation(temp_dir):
    splits_dir = os.path.join(temp_dir, "splits")
    cfg = SplitsConfig(
        datasets=["cora"],
        source="generate",
        root=DATA_DIR,
        splits_dir=splits_dir,
        folds=2,
        overwrite=True,
    )
    splits(cfg)
    for fold in range(2):
        assert os.path.exists(
            os.path.join(splits_dir, f"cora_split_0.6_0.2_{fold}.npz")
        )


@pytest.mark.integration
def test_sweep_integration(temp_dir):
    yaml_path = Path(temp_dir) / "test_sweep.yaml"
    sweep_data = {
        "model": "nsd",
        "dataset": {"name": "texas", "root": DATA_DIR},
        "search_space": {
            "stalk_dim": {"type": "int", "low": 2, "high": 3},
            "lr": {"type": "float", "low": 0.01, "high": 0.1},
        },
        "config": {
            "n_trials": 2,
            "study_name": "test-sweep",
            "n_seeds_per_trial": 1,
            "seed": 42,
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(sweep_data, f)

    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        sweep(yaml_path)
    finally:
        os.chdir(old_cwd)
    assert os.path.exists(os.path.join(temp_dir, "texas_nsd_bestconf.yaml"))
