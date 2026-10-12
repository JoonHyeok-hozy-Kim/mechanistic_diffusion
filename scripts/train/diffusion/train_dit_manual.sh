#!/bin/bash

PY_SCRIPT_PATH="train/diffusion/train_dit.py"
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

# Args : Two Dimensional
dataset_name=two_dimensional
channel_size=2
image_size=1
p=23
train_fraction=0.9
num_images_per_class=4
source_data_confidence_threshold=0.8
patch_size=1
model_size=L
prediction_target=x
noise_schedule=linear
# noise_schedule=vp
# noise_schedule=ve
depth=1
ffn_type=gelu
lr=3e-4
lr_schedule_type=ws
# lr_schedule_type=s
# lr_schedule_type=d
weight_decay=0.05
epochs=500000
batch_size=262144
timestep_sampling_type='normal'
sample_freq=5000
resume_wandb_url=false
# prev_lr_schedule_type='ws'
# prev_lr_schedule_type='s'
BOOLEAN_ARGS=(
    # --use_ffn_sandwich
    --use_layer_norm
    --use_rope_2d
    --use_ffn_residual
    # --dry_run
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
    --dataset_name "$dataset_name" \
    --channel_size "$channel_size" \
    --image_size "$image_size" \
    --p "$p" \
    --train_fraction "$train_fraction" \
    --num_images_per_class "$num_images_per_class" \
    --source_data_confidence_threshold "$source_data_confidence_threshold" \
    --patch_size "$patch_size" \
    --model_size "$model_size" \
    --prediction_target "$prediction_target" \
    --noise_schedule "$noise_schedule" \
    --depth "$depth" \
    --ffn_type "$ffn_type" \
    --lr "$lr" \
    --lr_schedule_type "$lr_schedule_type" \
    --weight_decay "$weight_decay" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --timestep_sampling_type "$timestep_sampling_type" \
    --sample_freq "$sample_freq" \
    --resume_wandb_url "$resume_wandb_url" \
    --prev_lr_schedule_type "$prev_lr_schedule_type" \
    "${BOOLEAN_ARGS[@]}"
    
echo "==================================="
echo "Ends at $(date)"