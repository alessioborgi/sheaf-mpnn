#!/bin/bash
# One Optuna sweep (250 trials, logged to W&B) for one dataset/model/restriction map.
#
# Usage (from the repo root; partition/account are cluster-specific):
#   sbatch -p <partition> -A <account> -J nsd-chameleon_filtered-orthogonal \
#     scripts/slurm/sweep.sbatch configs/chameleon_filtered/nsd/orthogonal.yaml
#
# The YAML path encodes the job: configs/<dataset>/<model>/<restriction_map>.yaml.
# Requires WANDB_API_KEY in the submitting environment.
#
#SBATCH --job-name=sheaf-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

YAML="${1:?usage: sbatch scripts/slurm/sweep.sbatch configs/<dataset>/<model>/<restriction_map>.yaml}"
N_TRIALS="${N_TRIALS:-250}"
WANDB_PROJECT="${WANDB_PROJECT:-sheaf-mpnn-runs}"
WANDB_ENTITY="${WANDB_ENTITY:-adavit}"

# sbatch copies this script to a spool dir, so resolve paths from the submit dir.
cd "${SLURM_SUBMIT_DIR:?must be submitted with sbatch}"
[ -f "$YAML" ] || { echo "no such config: $YAML" >&2; exit 1; }
: "${WANDB_API_KEY:?export WANDB_API_KEY before sbatch}"

# Each YAML header documents its preset, and preset names are not derivable from the path.
PRESET="$(grep -o -- '--preset [A-Za-z0-9_]*' "$YAML" | head -n1 | cut -d' ' -f2)"
[ -n "$PRESET" ] || { echo "no --preset found in the header of $YAML" >&2; exit 1; }

# Patched copy keeps the file stem, which sweep.py checks against the preset variant.
RUN_DIR="slurm_configs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
RUN_YAML="${RUN_DIR}/$(basename "$YAML")"
sed -E \
  -e "s/^([[:space:]]*)n_trials:.*/\1n_trials: ${N_TRIALS}/" \
  -e "s/^([[:space:]]*)#?[[:space:]]*wandb_project:.*/\1wandb_project: ${WANDB_PROJECT}/" \
  -e "s/^([[:space:]]*)#?[[:space:]]*wandb_entity:.*/\1wandb_entity: ${WANDB_ENTITY}/" \
  "$YAML" > "$RUN_YAML"
if ! grep -q "wandb_project:" "$RUN_YAML"; then
  sed -i -E "s/^config:[[:space:]]*$/config:\n  wandb_project: ${WANDB_PROJECT}\n  wandb_entity: ${WANDB_ENTITY}/" "$RUN_YAML"
fi
grep -q "wandb_project: ${WANDB_PROJECT}" "$RUN_YAML" \
  || { echo "could not enable W&B logging in $RUN_YAML" >&2; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== $(date -Is) job=${SLURM_JOB_ID} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "===== config=${YAML} preset=${PRESET} trials=${N_TRIALS} wandb=${WANDB_ENTITY}/${WANDB_PROJECT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --frozen sheaf sweep --yaml-path "$RUN_YAML" --preset "$PRESET"

echo "===== $(date -Is) done"
