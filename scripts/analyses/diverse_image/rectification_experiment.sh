#!/bin/bash

PY_SCRIPT_PATH="analyses/diverse_image/rectification_experiment.py"
OUT_DIR="logs/${PY_SCRIPT_PATH}"
mkdir -p "${OUT_DIR}"

DATE_WITH_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUT_DIR}/${DATE_WITH_TIME}_run_${SLURM_JOB_ID}.log"

exec > "$OUTPUT_FILE" 2>&1

echo "==================================="
date
echo "Job running on node: $(hostname)"
echo "==================================="

source ./venv/bin/activate
echo "[DEBUG] Python check:"
python --version

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"
echo "==================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "${FILE_NAME} starts at $(date)"
echo "==================================="

# Args : EMNIST Diverse Image
generative_model_path=weights/train_weights_reserved/emnist_diverse_image/wandb-260803_033356-w9hckjoc-dataset_name_emnist-p_23-train_fraction_0.9-num_images_per_class_256-patch_size_32-model_size_XL-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-lr_0.0005-lr_schedule_type_ws-weight_decay_0.01/lr_schedule_type_d/model_final.pth
dataset_name=emnist
channel_size=1
image_size=32
p=23
num_images_per_class=256
source_data_confidence_threshold=0.8
patch_size=32
model_size=XL
prediction_target=x
noise_schedule=linear
depth=1
ffn_type=gelu
sample_steps=50
BOOLEAN_ARGS=(
    --use_ffn_sandwich
    --use_layer_norm
    --use_rope_2d
    --use_ffn_residual
    # --correct_result_img
    --latex_formatted_out
)

python -u ${PY_SCRIPT_PATH} \
    --generative_model_path "$generative_model_path" \
    --dataset_name "$dataset_name" \
    --channel_size "$channel_size" \
    --image_size "$image_size" \
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

echo "-----------------------------------"

echo "==================================="
echo "Fin at $(date)"