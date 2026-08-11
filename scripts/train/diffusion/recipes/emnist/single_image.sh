#!/bin/bash

TRAIN_DIT_SCRIPT_PATH="scripts/train/diffusion/train_dit.sh"

if [ ! -x "${TRAIN_DIT_SCRIPT_PATH}" ]; then
  chmod +x "${TRAIN_DIT_SCRIPT_PATH}"
fi

"${TRAIN_DIT_SCRIPT_PATH}" \
    --dataset_name "emnist" \
    --channel_size 1 \
    --image_size 32 \
    --p 23 \
    --train_fraction 0.9 \
    --num_images_per_class 1 \
    --source_data_confidence_threshold 0.8 \
    --patch_size 32 \
    --model_size "L" \
    --prediction_target "x" \
    --noise_schedule "linear" \
    --depth 1 \
    --ffn_type "gelu" \
    --lr 5e-4 \
    --lr_schedule_type "ws" \
    --weight_decay 0.01 \
    --epochs 300000 \
    --batch_size 4096 \
    --timestep_sampling_type "normal" \
    --sample_freq 10000 \
    --use_layer_norm \
    --use_rope_2d \
    --use_ffn_residual
    # --dry_run