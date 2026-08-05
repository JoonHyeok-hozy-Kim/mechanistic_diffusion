# mechanistic_diffusion
A Mechanistic Study of Algorithmic Generalization in Diffusion Models



## Getting Started
### Inference Only

### Train model by yourself
1. Clone the repository and copy the example environment file:
   ```bash
   bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your own Wandb API key and entity.


#### Recipe
- EMNIST
  - $P=23, N=1$
    ```bash
    dataset_name=emnist
    p=23
    train_fraction=0.9
    num_images_per_class=1
    source_data_confidence_threshold=0.7
    patch_size=32
    model_size=L
    prediction_target=x
    depth=1
    ffn_type=gelu
    lr=3e-4
    lr_schedule_type=ws
    weight_decay=0.02
    epochs=200000
    batch_size=4096
    sample_freq=1000
    resume_wandb_url=false
    BOOLEAN_ARGS=(
       # --use_ffn_sandwich
       --use_layer_norm
       --use_rope_2d
       --use_ffn_residual
       # --dry_run
    )
    ```
  - $P=23, N=256$
    ```bash
    dataset_name=emnist
    p=23
    train_fraction=0.9
    num_images_per_class=256
    source_data_confidence_threshold=0.8
    patch_size=32
    model_size=XL
    prediction_target=x
    depth=1
    ffn_type=gelu
    lr=5e-4
    lr_schedule_type=ws
    weight_decay=0.01
    epochs=300000
    batch_size=4096
    sample_freq=5000
    resume_wandb_url=false
    BOOLEAN_ARGS=(
       --use_ffn_sandwich
       --use_layer_norm
       --use_rope_2d
       --use_ffn_residual
       # --dry_run
    )
    ```
- KMNIST
  - $P=47, N=256$
    ```bash
    dataset_name=kmnist
    p=47
    train_fraction=0.9
    num_images_per_class=256
    source_data_confidence_threshold=0.8
    patch_size=32
    model_size=XL
    prediction_target=x
    depth=1
    ffn_type=gelu
    lr=3e-4
    lr_schedule_type=ws
    weight_decay=0.01
    epochs=300000
    batch_size=4096
    sample_freq=5000
    resume_wandb_url=false
    BOOLEAN_ARGS=(
       --use_ffn_sandwich
       --use_layer_norm
       --use_rope_2d
       --use_ffn_residual
       # --dry_run
    )
    ```