#!/bin/bash

PY_SCRIPT_PATH="classifier_models/resnet/train_resnet_classifiers.py"
echo "PY_SCRIPT_PATH: $PY_SCRIPT_PATH"

LOG_DIR="./logs/${PY_SCRIPT_PATH%.py}"
echo "LOG_DIR: $LOG_DIR"
mkdir -p "${LOG_DIR}"

DATE_WITH_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${LOG_DIR}/${DATE_WITH_TIME}_run_${SLURM_JOB_ID:-local}.log"
exec > "$OUTPUT_FILE" 2>&1

echo "==================================="
date
echo "Job running on node: $(hostname)"
echo "==================================="

echo "[DEBUG] Python check:"
python --version

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"
echo "==================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1


# Arguments
data_source=emnist
train_fraction=0.8
BOOLEAN_ARGS=(
    --dry_run
)

echo "==================================="
if [[ " ${BOOLEAN_ARGS[@]} " =~ " --dry_run " ]]; then
    echo "Dry run mode detected. Skipping Wandb Settings."
    echo "Dry run on local"
else
    # exprt WANDB_API_KEY and WANDB_ENTITY from .env
    echo "Wandb Settings"
    set -a
    source .env
    set +a
    # echo "API Key : $WANDB_API_KEY"
    # echo "Entity  : $WANDB_ENTITY"
fi
echo "==================================="


echo "Starts at $(date)"
echo "==================================="

python -u "${PY_SCRIPT_PATH}" \
    --data_source "$data_source" \
    --train_fraction "$train_fraction" \
    "${BOOLEAN_ARGS[@]}"
    
echo "==================================="
echo "Ends at $(date)"