import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import cast

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
from exp.datasets.graph_classification import GCHetSheafDataModule
from exp.datasets.hgb import HetSheafDataModule
from exp.datasets.info import GCDatasetInfo, HetDatasetInfo, LPDatasetInfo
from exp.datasets.link_pred import LastFMDataModule
from exp.gc_module import GCLightningModule
from exp.gen_splits import SplitsConfig, splits
from exp.lp_module import LPLightningModule
from exp.module import HetSheafLightningModule, SheafLightningModule
from exp.sweeps.sweep import sweep

DATA_DIR = "exp/data"
MAX_EPOCHS = 3
SEED = 42


def _set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)


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
    epochs=MAX_EPOCHS,
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
            epochs=epochs,
            early_stopping=epochs + 1,
            batch_size=1,
        ),
    )


def _trainer(max_epochs=MAX_EPOCHS):
    return Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )


def _check_weight_updates(module, datamodule, max_epochs=MAX_EPOCHS):
    """Verify that model parameters are actually updated during training."""
    _set_seed()
    before_params = [p.clone().detach() for p in module.parameters()]

    t = _trainer(max_epochs)
    t.fit(module, datamodule)

    after_params = [p.detach() for p in module.parameters()]
    changed = any(
        not torch.equal(b, a) for b, a in zip(before_params, after_params, strict=False)
    )
    assert changed, "No parameters were updated during training! Check gradient flow."
    return t


def _train_and_assert(
    module, datamodule, metric: str, threshold: float, max_epochs=MAX_EPOCHS
) -> None:
    t = _check_weight_updates(module, datamodule, max_epochs)
    [res] = t.test(module, datamodule, verbose=False)
    assert res[metric] > threshold


def _apply_het_flags(cfg: Config, info: HetDatasetInfo | LPDatasetInfo) -> None:
    cfg.model.num_node_types = info.num_node_types
    cfg.model.num_edge_types = info.num_edge_types
    cfg.model.use_node_types = True
    cfg.model.use_edge_types = True


# When adding a new ModelType, add one (model_type, variant) entry to the
# appropriate list below. This ensures the full Config → registry →
# LightningModule → Trainer wiring is exercised for every model family.
HOMOGENEOUS_VARIANTS = [
    (ModelType.NSD, "diagonal"),
    (ModelType.NSD, "general"),
    (ModelType.NSD, "orthogonal"),
    (ModelType.NSD, "low_rank"),
    (ModelType.NSD, "general_attention"),
    (ModelType.NSD, "orthogonal_attention"),
    (ModelType.NSP, "diagonal"),
    (ModelType.NSP, "general"),
    (ModelType.NSP, "orthogonal"),
    (ModelType.NSP, "low_rank"),
    (ModelType.NSP, "general_attention"),
    (ModelType.NSP, "orthogonal_attention"),
    (ModelType.SHEAF_ATTN, "sheafan"),
    (ModelType.SHEAF_ATTN, "res_sheafan"),
]

HETEROGENEOUS_VARIANTS = [
    (ModelType.HET_NSD, "diagonal"),
    (ModelType.HET_NSD, "general"),
    (ModelType.HET_NSD, "orthogonal"),
    (ModelType.HET_NSD, "low_rank"),
    (ModelType.HET_NSD, "general_attention"),
    (ModelType.HET_NSD, "orthogonal_attention"),
    (ModelType.HET_NSP, "diagonal"),
    (ModelType.HET_NSP, "general"),
    (ModelType.HET_NSP, "orthogonal"),
    (ModelType.HET_NSP, "low_rank"),
    (ModelType.HET_NSP, "general_attention"),
    (ModelType.HET_NSP, "orthogonal_attention"),
    (ModelType.HET_SHEAF_ATTN, "sheafan"),
    (ModelType.HET_SHEAF_ADJ, "general"),
    (ModelType.HET_SHEAF_ADJ, "orthogonal"),
    (ModelType.HET_DNSD, "general"),
    (ModelType.HET_DNSD, "orthogonal"),
]


@pytest.mark.integration
@pytest.mark.parametrize("model_type,variant", HOMOGENEOUS_VARIANTS)
class TestHomogeneous:
    def test_nc(self, model_type, variant):
        dm = SheafDataModule("cora", root=DATA_DIR)
        dm.setup()
        cfg = _minimal_cfg(model_type=model_type, variant=variant)
        _train_and_assert(
            SheafLightningModule(cfg, dm.info),
            dm,
            "test_acc",
            1.0 / dm.info.num_classes,
        )

    def test_gc(self, model_type, variant):
        dm = GCHetSheafDataModule("mutag", root=DATA_DIR)
        dm.setup()
        cfg = _minimal_cfg(model_type=model_type, variant=variant)
        _train_and_assert(
            GCLightningModule(cfg, cast(GCDatasetInfo, dm.info)),
            dm,
            "test_acc",
            1.0 / dm.info.num_classes,
        )

    def test_lp(self, model_type, variant):
        dm = LastFMDataModule(
            data_dir=DATA_DIR, homogeneous=True, node_feature_type="degree"
        )
        dm.prepare_data()
        cfg = _minimal_cfg(
            model_type=model_type, variant=variant, normalize_output=False
        )
        cfg.model.jknet = True
        _train_and_assert(
            LPLightningModule(cfg, cast(LPDatasetInfo, dm.info)), dm, "test_auroc", 0.5
        )


@pytest.mark.integration
@pytest.mark.parametrize("model_type,variant", HETEROGENEOUS_VARIANTS)
class TestHeterogeneous:
    def test_nc(self, model_type, variant):
        dm = HetSheafDataModule("dblp", root=DATA_DIR)
        dm.setup()
        info = cast(HetDatasetInfo, dm.info)
        cfg = _minimal_cfg(dataset_name="dblp", model_type=model_type, variant=variant)
        _apply_het_flags(cfg, info)
        _train_and_assert(
            HetSheafLightningModule(cfg, info), dm, "test_acc", 1.0 / info.num_classes
        )

    def test_gc(self, model_type, variant):
        from exp.gc_module import HetGCLightningModule

        dm = GCHetSheafDataModule("mutag", root=DATA_DIR)
        dm.setup()
        info = cast(HetDatasetInfo, dm.info)
        cfg = _minimal_cfg(dataset_name="mutag", model_type=model_type, variant=variant)
        _apply_het_flags(cfg, info)
        _train_and_assert(
            HetGCLightningModule(cfg, cast(GCDatasetInfo, dm.info)),
            dm,
            "test_acc",
            1.0 / info.num_classes,
        )

    def test_lp(self, model_type, variant):
        from exp.lp_module import HetLPLightningModule

        dm = LastFMDataModule(
            data_dir=DATA_DIR, homogeneous=True, node_feature_type="degree"
        )
        dm.prepare_data()
        info = cast(LPDatasetInfo, dm.info)
        cfg = _minimal_cfg(
            dataset_name="lastfm",
            model_type=model_type,
            variant=variant,
            normalize_output=False,
        )
        _apply_het_flags(cfg, info)
        cfg.model.jknet = True
        _train_and_assert(HetLPLightningModule(cfg, info), dm, "test_auroc", 0.5)


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
