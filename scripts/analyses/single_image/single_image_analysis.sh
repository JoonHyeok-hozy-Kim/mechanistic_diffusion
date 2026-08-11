#!/bin/bash

PY_SCRIPT_PATH="analyses/singe_image/single_image_analysis.py"
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

# Args : EMNIST Single Image
generative_model_path=weights/train_weights_reserved/emnist_single_image/wandb-260802_052838-4xrmb5ax-dataset_name_emnist-p_23-train_fraction_0.9-num_images_per_class_1-patch_size_32-model_size_L-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-lr_0.0005-lr_schedule_type_ws-weight_decay_0.01/model_final.pth
dataset_name=emnist
channel_size=1
image_size=32
p=23
num_images_per_class=1
source_data_confidence_threshold=0.8
patch_size=32
model_size=L
prediction_target=x
noise_schedule=linear
depth=1
ffn_type=gelu
sample_steps=1
BOOLEAN_ARGS=(
    # --use_ffn_sandwich
    --use_layer_norm
    --use_rope_2d
    --use_ffn_residual
    # --dry_run
    --latex_formatted_out
    # --ffn_out_pre_act_only
)

echo "Starts at $(date)"
echo "==================================="

python -u "${PY_SCRIPT_PATH}" \
    --generative_model_path "$generative_model_path" \
    --dataset_name "$dataset_name" \
    --p "$p" \
    --num_images_per_class "$num_images_per_class" \
    --source_data_confidence_threshold "$source_data_confidence_threshold" \
    --patch_size "$patch_size" \
    --model_size "$model_size" \
    --prediction_target "$prediction_target" \
    --noise_schedule "$noise_schedule" \
    --depth "$depth" \
    --ffn_type "$ffn_type" \
    --sample_steps "$sample_steps" \
    "${BOOLEAN_ARGS[@]}"
    
echo "==================================="
echo "Ends at $(date)"