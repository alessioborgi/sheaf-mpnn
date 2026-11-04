#!/bin/bash
# Creates an isolated environment for external/neural-sheaf-diffusion,
# matching its pinned environment_gpu.yml (torch 1.11.0 / PyG 2.0.4) via pip
# wheels instead of conda. Only installs what
# src/exp/benchmark/original_entrypoint.py actually imports; skips wandb,
# GitPython, torchdiffeq, and pytest, which the original repo's own
# exp/run.py needs but our entrypoint never imports.
#
# Usage:
#   CUDA_TAG=cu113 bash scripts/benchmark/setup_original_env.sh   # GPU node
#   CUDA_TAG=cpu   bash scripts/benchmark/setup_original_env.sh   # no GPU
# Verify wheel availability for a different CUDA_TAG/Python combo at
# https://data.pyg.org/whl/ before changing it.

set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

CUDA_TAG="${CUDA_TAG:-cu113}"
REPO_DIR="external/neural-sheaf-diffusion"
VENV_DIR="$REPO_DIR/.venv"

uv venv --python 3.9 "$VENV_DIR"
PY="$VENV_DIR/bin/python"

uv pip install -p "$PY" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  torch==1.11.0

# The torch-1.11.0 wheel index carries no local version tags (+ptXXXcuYYY
# came later) and torch-sparse tops out at 0.6.13 for torch 1.11; the -f
# index alone selects the right CUDA build.
uv pip install -p "$PY" \
  "torch-scatter==2.0.9" \
  "torch-sparse==0.6.13" \
  -f "https://data.pyg.org/whl/torch-1.11.0+${CUDA_TAG}.html"

uv pip install -p "$PY" \
  torch-geometric==2.0.4 \
  networkx==2.6.3 \
  "numpy<2" \
  pyyaml

# torch-householder ships only an sdist that compiles a C++ extension, and
# its build requirement is an unpinned "torch>=1.5" -- under the default
# PEP-517 build isolation the extension gets compiled against the *latest*
# torch's C++ ABI and then fails to import under this venv's torch 1.11.
# Build without isolation so it compiles against the torch installed above
# (setuptools/wheel/ninja must be present in the venv for that).
# PyTorch 1.11 imports pkg_resources, removed in setuptools 82.
uv pip install -p "$PY" "setuptools<82" wheel ninja
uv pip install -p "$PY" --no-build-isolation torch-householder==1.0.1

echo "Original-repo env ready at $VENV_DIR"
