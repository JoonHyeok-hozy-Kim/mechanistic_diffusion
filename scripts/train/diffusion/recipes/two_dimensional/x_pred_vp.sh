#!/bin/bash

TRAIN_DIT_SCRIPT_PATH="scripts/train/diffusion/train_dit.sh"

if [ ! -x "${TRAIN_DIT_SCRIPT_PATH}" ]; then
  chmod +x "${TRAIN_DIT_SCRIPT_PATH}"
fi

"${TRAIN_DIT_SCRIPT_PATH}" \
    --dataset_name "two_dimensional" \
    --channel_size 2 \
    --image_size 1 \
    --p 23 \
    --train_fraction 0.9 \
    --num_images_per_class 4 \
    --source_data_confidence_threshold 0.8 \
    --patch_size 1 \
    --model_size "L" \
    --prediction_target "x" \
    --noise_schedule "vp" \
    --depth 1 \
    --ffn_type "gelu" \
    --lr 3e-4 \
    --lr_schedule_type "ws" \
    --weight_decay 0.05 \
    --epochs 500000 \
    --batch_size 262144 \
    --timestep_sampling_type "uniform_skewed" \
    --sample_freq 5000 \
    --use_layer_norm \
    --use_rope_2d \
    --use_ffn_residual
    # --dry_run