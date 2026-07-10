# Installation

`sheaf_mpnn` requires **Python >= 3.13** and uses [uv](https://github.com/astral-sh/uv) for dependency management.

## Base environment

```bash
git clone https://github.com/alessioborgi/pytorch-SheafNeuralNetworks
cd pytorch-SheafNeuralNetworks
uv sync
```

This installs both `sheaf_mpnn` (the core library) and `exp` (the experiment runner). Activate the venv, then verify:

```bash
python -c "import sheaf_mpnn; print(sheaf_mpnn.__version__)"
```

W&B logging and the Optuna-W&B sweep integration are part of the base
environment, so no extra install step is needed for sweeps.

## Optional groups

| Group        | Command                              | Provides                                              |
|--------------|--------------------------------------|-------------------------------------------------------|
| `dev` group  | `uv sync --dev`                      | tests, ruff, mypy, pre-commit                         |
| `docs` group | `uv sync --group docs`               | Sphinx, pydata-sphinx-theme, MyST, autodoc extensions |

The `docs` group is what CI uses to build this site; see
[the docs CI workflow](https://github.com/alessioborgi/pytorch-SheafNeuralNetworks/blob/main/.github/workflows/docs.yml).
