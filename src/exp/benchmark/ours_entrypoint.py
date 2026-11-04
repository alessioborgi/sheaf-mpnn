# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Single-fold / single-seed training run for the "ours" side of the
efficiency benchmark, instrumented with the same wall-clock and peak-GPU
memory tracking used by ``exp.sweeps.sweep``.

Usage
-----
    python -m exp.benchmark.ours_entrypoint \
        --dataset texas --variant orthogonal --mode efficiency --index 0 \
        --hparams configs/benchmark/bestconf/ours/texas/orthogonal.yaml \
        --epoch-cap 30 --warmup-epochs 5 \
        --out results/benchmark/nsd_vs_original/texas/orthogonal/ours/efficiency/0.json
"""

from __future__ import annotations

import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from exp.benchmark.hparams import load_ours_hparams
from exp.benchmark.jobspec import Mode
from exp.benchmark.result import ResultRecord
from exp.data import SheafDataModule
from exp.module import SheafLightningModule
from exp.resource import resource_metrics, start_resource_tracking


class _EpochTimer(Callback):
    """Records the wall-clock duration of every training epoch."""

    def __init__(self) -> None:
        self.epoch_times: list[float] = []
        self._epoch_start = 0.0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.epoch_times.append(time.perf_counter() - self._epoch_start)


def _steady_state_epoch_time(epoch_times: list[float], warmup_epochs: int) -> float:
    steady = epoch_times[warmup_epochs:] or epoch_times
    return sum(steady) / len(steady) if steady else 0.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(
    *,
    dataset: str,
    variant: str,
    mode: Mode,
    index: int,
    hparams_path: Path,
    cuda: int = 0,
    base_seed: int = 42,
    epoch_cap: int = 30,
    warmup_epochs: int = 5,
) -> ResultRecord:
    """Train+evaluate one fold (``mode="convergence"``) or one epoch-capped
    seed run on fold 0 (``mode="efficiency"``).
    """
    ours = load_ours_hparams(hparams_path)
    file_variant = str(ours.best_params["variant"])
    if file_variant != variant:
        # The model is built from best_params; the CLI --variant only labels
        # the result record. Left unchecked, a mislabeled hparams file would
        # silently benchmark the wrong model under this job's variant name.
        raise ValueError(
            f"--variant {variant!r} does not match best_params variant "
            f"{file_variant!r} in {hparams_path}"
        )
    fold = index if mode == "convergence" else 0
    _set_seed(base_seed + index)

    cfg = ours.to_config(dataset=dataset)
    dm = SheafDataModule(cfg.dataset.name, root=cfg.dataset.root, fold=fold)
    dm.setup()
    info = dm.info
    module = SheafLightningModule(cfg, info)
    monitor = "val_loss" if cfg.optim.stop_strategy == "loss" else f"val_{info.metric}"
    ckpt_mode = "min" if cfg.optim.stop_strategy == "loss" else "max"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [cuda] if torch.cuda.is_available() else "auto"

    inference_latency_ms: float | None = None

    if mode == "efficiency":
        timer = _EpochTimer()
        trainer = Trainer(
            max_epochs=epoch_cap,
            callbacks=[timer],
            logger=False,
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
            # Efficiency mode doesn't early-stop, so the validation loop is
            # dead weight -- and Lightning runs it *between*
            # on_train_epoch_start/end, which would put validation cost inside
            # the per-epoch timing. The "original" side times train_epoch()
            # only, so exclude validation here to keep the two symmetric.
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        start = start_resource_tracking(cuda)
        trainer.fit(module, dm)
        perf = resource_metrics(start, cuda, epochs=len(timer.epoch_times))
        perf["epoch_time_s"] = _steady_state_epoch_time(
            timer.epoch_times, warmup_epochs
        )
        [test_res] = trainer.test(module, dm, verbose=False)

        # Lightning's teardown moves the module back to the CPU after
        # fit/test, and the dataloader yields CPU batches, so both have to be
        # put back on the training device or this measures CPU latency on a
        # GPU node.
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        module.eval()
        module.to(device)
        batch = next(iter(dm.test_dataloader())).to(device)
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            module._forward(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_latency_ms = (time.perf_counter() - t0) * 1000
    else:
        with tempfile.TemporaryDirectory() as ckpt_dir:
            ckpt_cb = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=monitor,
                mode=ckpt_mode,
                save_top_k=1,
                filename="best",
            )
            trainer = Trainer(
                max_epochs=cfg.optim.epochs,
                callbacks=[
                    EarlyStopping(
                        monitor=monitor,
                        patience=cfg.optim.early_stopping,
                        mode=ckpt_mode,
                    ),
                    ckpt_cb,
                ],
                logger=False,
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
            )
            start = start_resource_tracking(cuda)
            trainer.fit(module, dm)
            perf = resource_metrics(start, cuda, epochs=trainer.current_epoch)
            [test_res] = trainer.test(module, dm, ckpt_path="best", verbose=False)

    param_count = sum(p.numel() for p in module.model.parameters())
    # KeyError over .get(..., 0.0): a metric-name mismatch should fail the
    # job, not record a silent 0.0 that skews the convergence comparison.
    test_metric = float(test_res[f"test_{info.metric}"])

    return ResultRecord(
        dataset=dataset,
        variant=variant,
        side="ours",
        mode=mode,
        index=index,
        fit_time_s=perf["fit_time_s"],
        epoch_time_s=perf["epoch_time_s"],
        epochs_trained=int(perf["epochs_trained"]),
        peak_gpu_mem_mb=perf.get("peak_gpu_mem_mb"),
        param_count=param_count,
        inference_latency_ms=inference_latency_ms,
        test_metric=test_metric,
        metric_name=info.metric,
    )


@dataclass
class _Args:
    dataset: str
    variant: str
    mode: Mode
    index: int
    hparams: Path
    out: Path
    cuda: int = 0
    base_seed: int = 42
    epoch_cap: int = 30
    warmup_epochs: int = 5


def main() -> None:
    args = tyro.cli(_Args)
    record = run_one(
        dataset=args.dataset,
        variant=args.variant,
        mode=args.mode,
        index=args.index,
        hparams_path=args.hparams,
        cuda=args.cuda,
        base_seed=args.base_seed,
        epoch_cap=args.epoch_cap,
        warmup_epochs=args.warmup_epochs,
    )
    record.write(args.out)


if __name__ == "__main__":
    main()
