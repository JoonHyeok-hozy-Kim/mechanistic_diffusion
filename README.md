# A Mechanistic Study of Algorithmic Generalization in Diffusion Models on Modular Addition

[![arXiv](https://img.shields.io/badge/arXiv-2604.17673-b31b1b.svg)](https://arxiv.org/abs/2604.17673)
[![ICLR DeLTa Workshop](https://img.shields.io/badge/ICLR%202026-DeLTa%20Workshop-blue.svg)](https://openreview.net/forum?id=rcpSrDLyBI)
[![HF Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/hozy-Joon/mechanistic_diffusion)

This is the official open source release of *A Mechanistic Study of Algorithmic Generalization in Diffusion Models*.

## Abstract
Despite their empirical success, how diffusion models generalize remains poorly understood from a mechanistic perspective. In this work, we provide a controlled mechanistic account of algorithmic learning in diffusion models. By utilizing the grokking phenomenon on an image-based modular addition task as a tractable testbed, we dissect the model's internal dynamics across two distinct visual regimes. In a [single-image regime](#analysis-1-single-image), mechanistic dissection reveals that the model implements modular addition by encoding operands as periodic representations and composing them at the self-attention--feedforward interface in a way that recovers the trigonometric addition identities. In a [diverse-image regime](#analysis-2-diverse-image-sampling-and-pca-visualization) with high intraclass variability, [we find that the iterative sampling process partitions the task into an arithmetic computation phase and a subsequent visual refinement phase](#analysis-3-incorrect-image-rectification-experiment-and-phase-transition-during-ode-sampling), with their transition precisely captured by an internal frequency-entropy signal. Together, our analysis clarifies the mechanics of algorithmic generalization of diffusion models in a controlled setting, while leaving open how broadly these mechanisms transfer beyond modular addition.



## Getting Started
### Environment Setup

```bash
# Clone the repository
git clone https://github.com/JoonHyeok-hozy-Kim/mechanistic_diffusion
cd mechanistic_diffusion

# Set up venv environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Weights
1. Download pretrained weights at [huggingface_hub](https://huggingface.co/hozy-Joon/mechanistic_diffusion)
   ```bash
   hf download hozy-Joon/mechanistic_diffusion weights/ --local-dir .
   ```
2. Or, you can train models by yourself.
   1. For the local run use the `--dry_run` option. Otherwise, setup `.env` as below.
      1. Clone the repository and copy the example environment file:
         ```bash
         bash
         cp .env.example .env
         ```
      2. Open `.env` and fill in your own Wandb API key and entity.
   2. Suggested recipes
      - EMNIST ($32\times 32$, $S=32$)
        - Single Image $(N=1)$
          ```bash
          bash scripts/train/diffusion/recipes/emnist/single_image.sh
          ```
        - Diverse Image $(N=256)$
          ```bash
          bash scripts/train/diffusion/recipes/emnist/diverse_image.sh
          ```
      - KMNIST ($32\times 32$, $S=32$)
        - Diverse Image $(N=256)$
          ```bash
          bash scripts/train/diffusion/recipes/kmnist/diverse_image.sh
          ```
      - 2D dataset ($S=1$)
         - $x$-prediction
           - linear noise schedule
             ```bash
             bash scripts/train/diffusion/recipes/two_dimensional/  x_pred_linear.sh
             ```
           - VP noise schedule
             ```bash
             bash scripts/train/diffusion/recipes/two_dimensional/x_pred_vp.sh
             ```
           - VE noise schedule
             ```bash
             bash scripts/train/diffusion/recipes/two_dimensional/x_pred_ve.sh
             ```
         - $v$-prediction (linear noise schedule)
           ```bash
           bash scripts/train/diffusion/recipes/two_dimensional/v_pred_linear.sh
           ```
         - $\epsilon$-prediction (linear noise schedule)
           ```bash
           bash scripts/train/diffusion/recipes/two_dimensional/epsilon_pred_linear.sh
           ```

### Analysis 1: Single Image
- Description: Calculate the fractional variances explained (FVEs) from the layer activations, visualize them with bar charts and heatmaps, and derive the trigonometry reconstruction table for each frequency.
- How to run
  ```bash
  # Make sure to specify the weight directory and the corresponding argument values
  bash scripts/analyses/single_image/single_image_analysis.sh
  ```
- Example) 
  - FFN's Pre-Activation Layer's result position $c$
    |FVE Barchart|FVE Heatmap|
    |:-:|:-:|
    |![](results/analyses/single_image/single_image_analysis/260810_203235-dataset_name_emnist-p_23-num_images_per_class_1-patch_size_32-model_size_L-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-sample_steps_1/fve_plots/L20-layers.0.feed_forward_out.hook_pre_activation/Grid_bar_two_degree_neuron_level_var_c.png)|![](results/analyses/single_image/single_image_analysis/260810_203235-dataset_name_emnist-p_23-num_images_per_class_1-patch_size_32-model_size_L-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-sample_steps_1/fve_plots/L20-layers.0.feed_forward_out.hook_pre_activation/Grid_heatmap_neuron_level_var_c.png)|
  - Trigonometry reconstruction Table
      |$W_L$|$u_k^\top \text{MLP}(a,b) \text{ and } v_k^\top \text{MLP}(a,b)$|FVE|
      |:-:|:-:|:-:|
      |$\cos(w_{1})$|$3,822.07\cos(w_{1} a) \cos(w_{1} b)-1,860.31\sin(w_{1} a) \sin(w_{1} b)$|0.01|
      |$\sin(w_{1})$|$1,979.02\cos(w_{1} a) \sin(w_{1} b)+1,789.55\sin(w_{1} a) \cos(w_{1} b)$|0.01|
      |$\cos(w_{2})$|$159.16\cos(w_{2} a) \cos(w_{2} b)-341.76\sin(w_{2} a) \sin(w_{2} b)$|0.00|
      |$\sin(w_{2})$|$785.35\cos(w_{2} a) \sin(w_{2} b)+779.45\sin(w_{2} a) \cos(w_{2} b)$|0.00|
      |$\cos(w_{3})$|$5,481.82\cos(w_{3} a) \cos(w_{3} b)-8,028.64\sin(w_{3} a) \sin(w_{3} b)$|0.02|
      |$\sin(w_{3})$|$6,078.35\cos(w_{3} a) \sin(w_{3} b)+5,524.31\sin(w_{3} a) \cos(w_{3} b)$|0.01|
      |$\cos(w_{4})$|$50,980.94\cos(w_{4} a) \cos(w_{4} b)-41,456.63\sin(w_{4} a) \sin(w_{4} b)$|0.26|
      |$\sin(w_{4})$|$17,386.96\cos(w_{4} a) \sin(w_{4} b)+15,832.02\sin(w_{4} a) \cos(w_{4} b)$|0.10|
      |$\cos(w_{5})$|$2,384.40\cos(w_{5} a) \cos(w_{5} b)-4,881.94\sin(w_{5} a) \sin(w_{5} b)$|0.01|
      |$\sin(w_{5})$|$5,924.81\cos(w_{5} a) \sin(w_{5} b)+6,162.13\sin(w_{5} a) \cos(w_{5} b)$|0.01|
      |$\cos(w_{6})$|$500.61\cos(w_{6} a) \cos(w_{6} b)-675.49\sin(w_{6} a) \sin(w_{6} b)$|0.00|
      |$\sin(w_{6})$|$695.69\cos(w_{6} a) \sin(w_{6} b)+673.62\sin(w_{6} a) \cos(w_{6} b)$|0.00|
      |$\cos(w_{7})$|$3,521.40\cos(w_{7} a) \cos(w_{7} b)-3,724.15\sin(w_{7} a) \sin(w_{7} b)$|0.01|
      |$\sin(w_{7})$|$9,258.93\cos(w_{7} a) \sin(w_{7} b)+8,584.56\sin(w_{7} a) \cos(w_{7} b)$|0.01|
      |$\cos(w_{8})$|$629,458.69\cos(w_{8} a) \cos(w_{8} b)-610,354.81\sin(w_{8} a) \sin(w_{8} b)$|0.87|
      |$\sin(w_{8})$|$570,603.19\cos(w_{8} a) \sin(w_{8} b)+571,393.38\sin(w_{8} a) \cos(w_{8} b)$|0.88|
      |$\cos(w_{9})$|$491,625.06\cos(w_{9} a) \cos(w_{9} b)-474,271.31\sin(w_{9} a) \sin(w_{9} b)$|0.86|
      |$\sin(w_{9})$|$502,036.81\cos(w_{9} a) \sin(w_{9} b)+513,029.75\sin(w_{9} a) \cos(w_{9} b)$|0.88|
      |$\cos(w_{10})$|$367,634.47\cos(w_{10} a) \cos(w_{10} b)-302,547.78\sin(w_{10} a) \sin(w_{10} b)$|0.80|
      |$\sin(w_{10})$|$237,343.55\cos(w_{10} a) \sin(w_{10} b)+237,641.44\sin(w_{10} a) \cos(w_{10} b)$|0.78|
      |$\cos(w_{11})$|$230,252.03\cos(w_{11} a) \cos(w_{11} b)-180,897.47\sin(w_{11} a) \sin(w_{11} b)$|0.73|
      |$\sin(w_{11})$|$213,711.89\cos(w_{11} a) \sin(w_{11} b)+218,311.89\sin(w_{11} a) \cos(w_{11} b)$|0.76|

<br>

### Analysis 2: Diverse Image Sampling and PCA Visualization
- Description: ODE sample the image, trace the FVE values, and visualize the periodic structure using PCA.
- How to run
    ```bash
    bash scripts/analyses/diverse_image/sample_and_plot.sh
    ```
- Examples)
  - Sample and plot FVE barcharts   
    ![](results/analyses/diverse_image/sample_and_plot/260811_041051-dataset_name_emnist-p_23-num_images_per_class_256-patch_size_32-model_size_XL-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-sample_steps_50/plot_fve/patch_0/00_animation.gif)
  - PCA visualization on result position $c$
    ![](results/analyses/diverse_image/sample_and_plot/260811_041051-dataset_name_emnist-p_23-num_images_per_class_256-patch_size_32-model_size_XL-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-sample_steps_50/plot_pca/variable_c/patch_0/00_animation.gif)

<br>

### Analysis 3: Incorrect Image Rectification Experiment and Phase Transition during ODE Sampling
- Description
  - Provide an incorrect image at the result position with varying noise levels, and trace the model's correction capability and the FVE entropy following the ODE trajectory.
  - We observe two phases during the ODE sampling procedure:
    1. Arithmetic Reasoning Phase: Maintains a low FVE entropy level indicating the internal periodic structure of the model, and succeeds in rectification.
    2. Visual Refinement Phase : FVE entropy increases, and the model fails to fix the incorrect result image.
  - The second derivative of the FVE entropy predicts the model's rectification failure timestep.
- How to run
   ```bash
   bash scripts/analyses/diverse_image/rectification_experiment.sh
   ```
- Examples)   
  ![](results/analyses/diverse_image/rectification_experiment/260811_015952-dataset_name_emnist-p_23-num_images_per_class_256-patch_size_32-model_size_XL-prediction_target_x-noise_schedule_linear-depth_1-ffn_type_gelu-sample_steps_50/p_23-emnist-incorrect/entropy_accuracy_dynamics_fourier_fve.png)


<br>

## Citation
```bibtex
@misc{kim2026grokkingdiffusionmodelscase,
      title={Grokking of Diffusion Models: Case Study on Modular Addition}, 
      author={Joon Hyeok Kim and Yong-Hyun Park and Mattis Dalsætra Østby and Jiatao Gu},
      year={2026},
      eprint={2604.17673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.17673}, 
}
```