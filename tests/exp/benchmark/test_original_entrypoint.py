from __future__ import annotations

import json

import torch
from torch import nn

from exp.benchmark.original_entrypoint import _steady_state_epoch_time, run_fold


class _FakeModel(nn.Module):
    """Stand-in for the original repo's DiscreteXSheafDiffusion models:
    exposes grouped_parameters() and ignores edge structure in forward().
    """

    def __init__(self, edge_index, args):
        super().__init__()
        self.sheaf_learner = nn.Linear(args["input_dim"], 8)
        self.lin = nn.Linear(8, args["output_dim"])

    def forward(self, x):
        return torch.log_softmax(self.lin(torch.relu(self.sheaf_learner(x))), dim=1)

    def grouped_parameters(self):
        sheaf = [p for n, p in self.named_parameters() if "sheaf_learner" in n]
        other = [p for n, p in self.named_parameters() if "sheaf_learner" not in n]
        return sheaf, other


class _FakeDataset:
    num_features = 5
    num_classes = 3


def _fake_data(n: int = 20):
    class _Data:
        pass

    data = _Data()
    data.x = torch.randn(n, 5)  # ty: ignore[unresolved-attribute]
    data.y = torch.randint(0, 3, (n,))  # ty: ignore[unresolved-attribute]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[: n // 2] = True
    data.train_mask = mask  # ty: ignore[unresolved-attribute]
    data.val_mask = mask  # ty: ignore[unresolved-attribute]
    data.test_mask = mask  # ty: ignore[unresolved-attribute]
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.edge_index = edge_index  # ty: ignore[unresolved-attribute]
    return data


def test_steady_state_epoch_time_discards_warmup():
    assert _steady_state_epoch_time([10.0, 5.0, 1.0, 1.0, 1.0], warmup_epochs=2) == 1.0


def _make_counting_model(calls: list[str]):
    """_FakeModel subclass that records "train" or "eval" for every forward
    pass, keyed off ``self.training`` (set by the standard nn.Module
    ``.train()``/``.eval()`` calls). Lets a test count eval-mode forward
    passes without needing a handle to the model instance run_fold creates
    internally.
    """

    class _CountingModel(_FakeModel):
        def forward(self, x):
            calls.append("train" if self.training else "eval")
            return super().forward(x)

    return _CountingModel


def test_run_fold_efficiency_mode_evaluates_once_not_per_epoch():
    # Regression for the bug where efficiency mode called evaluate() inside
    # the per-epoch loop: that leaked validation/test cost into fit_time_s,
    # biasing the ours-vs-original headline comparison. Efficiency mode
    # should only ever produce two eval-mode forward passes in total,
    # regardless of epoch_cap: the single post-loop evaluate() (for
    # test_metric) and the single post-loop inference-latency measurement.
    # A per-epoch evaluate() bug would instead produce epoch_cap + 1 of them,
    # and asserting on epochs_trained alone would not catch this.
    calls: list[str] = []
    hparams = {
        "lr": 0.01,
        "weight_decay": 0.0,
        "sheaf_decay": None,
        "early_stopping": 1,
        "stop_strategy": "loss",
    }
    result = run_fold(
        model_cls=_make_counting_model(calls),
        torch_module=torch,
        dataset=_FakeDataset(),
        data=_fake_data(),
        hparams=hparams,
        mode="efficiency",
        epoch_cap=6,
        warmup_epochs=2,
        device="cpu",
    )
    assert result["epochs_trained"] == 6
    assert calls.count("train") == 6
    assert calls.count("eval") == 2


def test_run_fold_efficiency_mode_runs_exactly_epoch_cap():
    hparams = {
        "lr": 0.01,
        "weight_decay": 0.0,
        "sheaf_decay": None,
        "early_stopping": 1,
        "stop_strategy": "loss",
    }
    result = run_fold(
        model_cls=_FakeModel,
        torch_module=torch,
        dataset=_FakeDataset(),
        data=_fake_data(),
        hparams=hparams,
        mode="efficiency",
        epoch_cap=6,
        warmup_epochs=2,
        device="cpu",
    )
    assert result["epochs_trained"] == 6
    assert result["inference_latency_ms"] is not None
    assert result["param_count"] > 0


def test_run_fold_works_with_architecture_only_hparams():
    # Regression: real "matched architecture" config files only carry d /
    # hidden_channels / layers; every optimisation flag has to come from
    # DEFAULT_ARGS or run_fold raises KeyError.
    result = run_fold(
        model_cls=_FakeModel,
        torch_module=torch,
        dataset=_FakeDataset(),
        data=_fake_data(),
        hparams={"d": 2, "hidden_channels": 4, "layers": 1},
        mode="efficiency",
        epoch_cap=3,
        warmup_epochs=1,
        device="cpu",
    )
    assert result["epochs_trained"] == 3
    assert result["inference_latency_ms"] is not None


def test_run_fold_convergence_mode_defaults_epoch_budget_from_default_args():
    # Convergence mode reads its epoch budget from args, which must fall back
    # to DEFAULT_ARGS["epochs"] when the hparams file omits it. early_stopping
    # is deliberately left to its default too.
    result = run_fold(
        model_cls=_FakeModel,
        torch_module=torch,
        dataset=_FakeDataset(),
        data=_fake_data(),
        hparams={"d": 2, "hidden_channels": 4, "layers": 1, "epochs": 4},
        mode="convergence",
        epoch_cap=30,
        warmup_epochs=5,
        device="cpu",
    )
    assert 0 < result["epochs_trained"] <= 4


def test_write_result_atomically_round_trips(tmp_path):
    from exp.benchmark.original_entrypoint import write_result_atomically

    path = tmp_path / "nested" / "0.json"
    write_result_atomically(path, {"dataset": "texas", "index": 0})
    assert json.loads(path.read_text()) == {"dataset": "texas", "index": 0}
    assert not (tmp_path / "nested" / "0.json.tmp").exists()


def test_run_fold_convergence_mode_respects_epoch_budget():
    hparams = {
        "lr": 0.01,
        "weight_decay": 0.0,
        "sheaf_decay": None,
        "early_stopping": 2,
        "stop_strategy": "loss",
        "epochs": 10,
    }
    result = run_fold(
        model_cls=_FakeModel,
        torch_module=torch,
        dataset=_FakeDataset(),
        data=_fake_data(),
        hparams=hparams,
        mode="convergence",
        epoch_cap=30,
        warmup_epochs=5,
        device="cpu",
    )
    assert 0 < result["epochs_trained"] <= 10
    assert result["inference_latency_ms"] is None
