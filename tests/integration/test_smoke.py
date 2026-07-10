# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""End-to-end smoke tests covering the three public surfaces.

Exercises the README Quick Start snippets, the exp.run CLI, and the
sweep CLI.  Requires network access on first run to download Cora splits.

Run with:
    uv run pytest tests/integration/ -v
"""

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.integration


class TestLibraryUsage:
    """README Quick Start snippets work verbatim."""

    def test_nsd_model_forward(self):
        from sheaf_mpnn.nsd import NSDModel, NSDVariant

        model = NSDModel(
            in_channels=1433,
            out_channels=7,
            stalk_dim=4,
            hidden_dim=16,
            num_layers=2,
            variant=NSDVariant.GENERAL,
            alpha=1.0,
        )
        n = 10
        x = torch.randn(n, 1433)
        edge_index = torch.randint(0, n, (2, 30))
        out = model(x, edge_index)
        assert out.shape == (n, 7)

    def test_general_nsd_conv_layer(self):
        from torch import nn

        from sheaf_mpnn.nsd import GeneralNSDConv

        d, hidden_dim = 4, 16
        encoder = nn.Linear(1433, d * hidden_dim)
        layer = GeneralNSDConv(
            stalk_dim=d,
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            context_dim=d * hidden_dim,
            alpha=1.0,
        )
        n = 10
        x = torch.randn(n, 1433)
        edge_index = torch.randint(0, n, (2, 30))
        x_stalk = encoder(x).view(-1, d, hidden_dim)
        x_feat = x_stalk.reshape(x_stalk.size(0), -1)
        x_stalk = layer(x_feat, x_stalk, edge_index)
        assert x_stalk.shape == (n, d, hidden_dim)


class TestExpRunCLI:
    """python -m exp.run works with a preset."""

    def test_preset_cora_runs(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "exp.run",
                "--preset",
                "cora",
                "--optim.epochs",
                "2",
                "--optim.early-stopping",
                "999",
                "--cv.folds",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        combined = result.stdout + result.stderr
        assert "acc" in combined.lower()

    def test_unknown_preset_exits_cleanly(self):
        result = subprocess.run(
            [sys.executable, "-m", "exp.run", "--preset", "does_not_exist"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "does_not_exist" in (result.stdout + result.stderr)


class TestSweepCLI:
    """python -m exp.sweeps.sweep runs a trial end-to-end."""

    def test_sweep_one_trial(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            model: nsd
            search_space:
              stalk_dim:
                type: int
                low: 2
                high: 4
              variant:
                type: categorical
                choices: [diagonal, general]
            config:
              n_trials: 1
              study_name: smoke-test
        """)
        (tmp_path / "sweep.yaml").write_text(yaml_content)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "exp.sweeps.sweep",
                "--yaml-path",
                str(tmp_path / "sweep.yaml"),
                "--preset",
                "cora",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "Best trial" in result.stdout
