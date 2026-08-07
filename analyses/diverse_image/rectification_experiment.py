import argparse
import os
import math
import gc
from datetime import datetime as dt
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.distributions as dist
import torchvision.utils as tvu

from utils.manage_project import get_config_str_dict, make_result_dir
from utils.seed import seed_everything
from models.dit import DiT_Llama
from models.rf import RF
from data.manage_dataset import get_confident_image_data_dict
from data.modular_addition import Inference_All_Image_Modular_Arithmetic_Dataset
from classifier_models.manage_classifiers import get_classifier
from analyses.utils.fourier_analysis import get_fourier_basis_and_names, get_indices_all_frequencies, \
    get_d2_indices_by_frequency
from analyses.utils.plot import plot_pca_all_layers, plot_fve_barchart_and_heatmap, generate_gif_animation

RANDOM_SEED = 46


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--generative_model_path", type=str, default="weights/")
        parser.add_argument("--classifier_weights_path", type=str, default="weights/classifier_resnet_reserved")
        parser.add_argument("--output_dir", type=str, default="results/")
        parser.add_argument("--dataset_name", type=str, default="emnist", help="Dataset type. e.g. emnist, kmnist, two_dimensional")
        parser.add_argument("--p", type=int, default=97, help="Modulus for modular arithmetic (should be <= 97 for all characters: A-Z, a-z, 0-9, special chars)")
        parser.add_argument("--num_images_per_class", type=int, default=256)
        parser.add_argument("--source_data_confidence_threshold", type=float, default=0.9, help="Confidence threshold for adding source image to the dataset")       
        parser.add_argument("--patch_size", type=int, default=2, help="Patch size for the model")
        parser.add_argument("--model_size", type=str, default="XL", help="S, M, L, XL")
        parser.add_argument("--prediction_target", type=str, choices=["x", "v", "epsilon"], help="DiT's prediction target provided to RF")
        parser.add_argument("--noise_schedule", type=str, default="linear", help="RF noise schedule")
        parser.add_argument("--depth", type=int, default=10)
        parser.add_argument("--ffn_type", type=str, default="gelu", choices=["swiglu", "gelu"], help="Choose FFN type : swiglu, gelu")
        parser.add_argument("--use_ffn_sandwich", action="store_true", help="Use prepending ffn to self-attention")
        parser.add_argument("--use_layer_norm", action="store_true", help="Apply LayerNorm in Transformer and FinalLayer blocks")
        parser.add_argument("--use_rope_2d", action="store_true", help="Apply rope 2d on Attention layers")
        parser.add_argument("--use_ffn_residual", action="store_true", help="Use residual connection at FFN")
        parser.add_argument("--sample_steps", type=int, default=50)
        parser.add_argument("--correct_result_img", action="store_true", help="Correct result image at position c")
        # parser.add_argument("--include_activations", action="store_true", help="")
        # parser.add_argument("--include_svd_statistics", action="store_true", help="")
        # parser.add_argument("--include_fve_statistics", action="store_true", help="")
        # parser.add_argument("--include_ode_statistics", action="store_true", help="")
        # parser.add_argument("--include_sample_images", action="store_true", help="")
        parser.add_argument("--latex_formatted_out", action="store_true", help="Latex formatted output")
        return parser.parse_args()
    args = parse_args()    
    
    now_str = dt.now().strftime("%y%m%d_%H%M")    
    script_dir_str = str(os.path.relpath(os.path.abspath(__file__))).split(".")[0]
    script_dir_list = list(script_dir_str.split("/"))
    script_file_name = os.path.basename(__file__).split(".")[0]
    config_dict, config_str = get_config_str_dict(args)
        
    wandb_exp_name = "-".join(script_dir_list)
    wandb_exp_name += f"-{script_file_name}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(RANDOM_SEED)
        
    # Directory settings
    base_result_dir = os.path.join(args.output_dir, script_dir_str)
    result_dir = make_result_dir(base_result_dir, config_str)    
    
    # Following GrokkingTransformer hyperparameters: d_model=128, num_heads=4, d_ff=512
    if args.model_size == "S":
        kwargs = dict(dim=128, n_layers=args.depth, n_heads=4, ffn_dim=512)
    elif args.model_size == "L":
        kwargs = dict(dim=512, n_layers=args.depth, n_heads=16, ffn_dim=2048)
    elif args.model_size == "XL":
        kwargs = dict(dim=2048, n_layers=args.depth, n_heads=64, ffn_dim=8192)
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")
    
    common_kwargs = dict(
        prediction_target=args.prediction_target, ffn_type=args.ffn_type, use_ffn_sandwich=args.use_ffn_sandwich, ffn_dim_multiplier=1,
        use_layer_norm=args.use_layer_norm, use_rope_2d=args.use_rope_2d, use_ffn_residual=args.use_ffn_residual, 
    )
    kwargs.update(common_kwargs)
            
    model = DiT_Llama(in_channels=1, input_size=32, patch_size=args.patch_size, **kwargs).cuda()   
    
    # Load trained weights
    print(f"Loading model weights from {args.generative_model_path}")
    try:
        loaded_weights = torch.load(args.generative_model_path, weights_only=False)
        if 'model_state_dict' in loaded_weights:
            model_weights = loaded_weights['model_state_dict']
            print(f"Weight with train/valid accuracy : {loaded_weights['max_train_accuracy']:.2f}/{loaded_weights['max_valid_accuracy']:.2f}")
        else:
            model_weights = loaded_weights
    except:
        raise ReferenceError(f"Failed loading weights")
    
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")
    
    # Cache all the activations
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
        
    rf = RF(model, time_sampling_type="uniform_skewed", noise_schedule=args.noise_schedule)
        
    classifier = get_classifier(dataset_name=args.dataset_name).to(device)    
    source_img_data_dict = get_confident_image_data_dict(
        args.dataset_name, args.p, args.num_images_per_class, 
        classifier, args.source_data_confidence_threshold,
    )
    
    ds_all = Inference_All_Image_Modular_Arithmetic_Dataset(
        p=args.p, num_images_per_class=args.num_images_per_class, 
        img_data_dict=source_img_data_dict, correct_result_img=args.correct_result_img
    )
    len_data = len(ds_all.data)
    batch_repeat = ds_all.batch_repeat
    dl_all = DataLoader(ds_all, batch_size=len_data, shuffle=False, drop_last=False, num_workers=0)    
    
    variables = ['c', 'a', 'b']
    num_variables = len(variables)
    
    # Patch related parameters
    input_size = model.input_size
    patch_size = model.patch_size
    grid_size = input_size // patch_size
    num_patches = grid_size ** 2
    patch_indices_by_variable = []
    for i in range(num_variables):
        indices = []
        for j in range(grid_size):
            var1 = i * grid_size + j * num_variables * grid_size
            for k in range(grid_size):
                var2 = var1 + k
                indices.append(var2)
        patch_indices_by_variable.append(indices)
            
    # Load Data : Ordered by (P_a, P_b) N^2 pairs in one batch
    dl_all_iter = iter(dl_all)
    answer_img, x_src, label_tgt, label_a, label_b = next(dl_all_iter)    
    
    # Sample from the operands dataset.
    x_src = x_src.to(device)
    answer_img = answer_img.to(device)
    fixed_noise = torch.randn_like(answer_img, device=device)
    answer_label = ((label_a + label_b) % args.p).to(device)
    
    # Fourier Analysis settings
    fourier_basis_d1, fourier_basis_names = get_fourier_basis_and_names(args.p, latex=args.latex_formatted_out)
    num_freqs = (args.p - 1) // 2
    num_2d_bases = num_variables**2 - 1    
    # Get idices of 1-degree bases and 2-degree bases with same frequences
    two_degree_idxes, two_degree_idxes_descs = get_indices_all_frequencies(args.p, latex=args.latex_formatted_out)  # (num_ks, num_2d_bases)  e.g. (11, 8) 
    indices_flattened = two_degree_idxes.flatten()  
    desc_flattened = two_degree_idxes_descs.flatten()    
    
    B = answer_img.size(0)
    dt_scalar = 1.0 / args.sample_steps
    dt_tensor = torch.tensor([dt_scalar] * B).to(device).view([B, *([1] * len(answer_img.shape[1:]))])
        
    # Settings dependent on is_correct_img
    correct_label_txt = "correct" if args.correct_result_img else "incorrect"
    is_correct_dir = os.path.join(result_dir, f"p_{args.p}-{args.dataset_name}-{correct_label_txt}")
    img_dir = os.path.join(is_correct_dir, "images")
    os.makedirs(img_dir, exist_ok=True) 
    save_img_idx = 6
    save_img_label = answer_label[save_img_idx]
    save_img_list = []
    
    target_layer_name = 'layers.0.feed_forward_out.hook_pre_activation'
    optimal_fves = [None] * args.sample_steps
    ode_accuracies = [None] * args.sample_steps
        
    with torch.no_grad():
        for start_t in range(args.sample_steps):
            print(f"[p={args.p} {args.dataset_name} {correct_label_txt}] Generating {start_t+1:02d}/{args.sample_steps:02d}")

            cache = {}
            model.remove_all_hooks()
            model.cache_all(cache)
        
            curr_sample_step = args.sample_steps - start_t
            curr_noise_ratio = curr_sample_step / args.sample_steps
            
            x_tgt = (1-curr_noise_ratio) * answer_img + curr_noise_ratio * fixed_noise
            x_tgt = x_tgt.to(device)
        
            save_img_path = os.path.join(img_dir, f"label_{save_img_label}-start_t_{start_t}-initial.png")
            img_tenser_cpu = x_tgt[save_img_idx].detach().cpu().clone()
            save_img_list.append((save_img_path, img_tenser_cpu))

            for sample_t in range(curr_sample_step, 0, -1):
                t_scalar = sample_t / args.sample_steps            
                t_tensor = torch.tensor([t_scalar] * B).to(x_tgt.device)

                vc = model(x_tgt, t_tensor, x_src)
                x_tgt = x_tgt - dt_tensor * vc 
                
                # Calculate optimal FVE only
                if sample_t == curr_sample_step:
                    for layer_name, activation in cache.items():
                        if layer_name != target_layer_name:
                            continue
                        
                        original_shape = activation.shape
                        assert len(original_shape) == 3, f"Unexpected activation shape of {layer_name} : {original_shape}"
                        assert original_shape[0] == batch_repeat*(args.p**2), f"Unexpected ({layer_name})[0] : {batch_repeat*(args.p**2)} (expected (batch_repeat)*p^2) vs {original_shape[0]} (actual)"
                        assert original_shape[1] == num_variables*num_patches, f"Unexpected ({layer_name})[1] : {num_variables*num_patches} (expected num_tokens*num_patches) vs {original_shape[1]} (actual)"
                        
                        hidden_dim = original_shape[2]
                        
                        # Batch Activation : Activation (for PCA), Singular Values           
                        batch_activation = activation.detach().clone()
                        batch_activation = batch_activation.reshape(batch_repeat * (args.p**2), num_patches * num_variables, hidden_dim)    # (B*P_a*P_b, -1)
                    
                        # Batch Average Activation : Fourier Analysis
                        mean_activation = activation.detach().clone()
                        mean_activation = mean_activation.reshape(batch_repeat, args.p**2, -1)     
                        mean_activation = torch.mean(mean_activation, dim=0)        
                        mean_activation = mean_activation.view(args.p, args.p, num_patches * num_variables, hidden_dim)
                        
                        for vdx, var in enumerate(variables):         
                            # As we target FFN-out, we focus only on the 'c' position.
                            if var != 'c':
                                continue
                                        
                            curr_var_patch_indices = patch_indices_by_variable[vdx]                        
                            batch_var_activation = batch_activation[:, curr_var_patch_indices, :]    # (B*P_u*P_v, num_patches, D)
                            mean_var_activation = mean_activation[:, :, curr_var_patch_indices, :]   # (P_a, P_b, num_patches, D)  
                            
                            # Centered Activation for PCA
                            batch_var_mean = batch_var_activation.mean(dim=0)               # (num_patches, D)
                            batch_var_centered = batch_var_activation - batch_var_mean      # (B*P_u*P_v, num_patches, D)
                            batch_var_centered = batch_var_centered.transpose(0, 1)         # (num_patches, B*P_u*P_v, D)
                            del batch_var_mean 
                        
                            # # Singular Value Squared
                            # if args.include_svd_statistics:                                          
                            #     singular_values_sq = torch.linalg.svdvals(batch_token_centered)     # (num_patches, min(B*P_u*P_v, D))
                            #     singular_values_sq = singular_values_sq ** 2
                            #     h5f_sv_sq.create_dataset(
                            #         f"{start_t:02d}", data=singular_values_sq.cpu().numpy(),
                            #         compression="gzip", compression_opts=4,
                            #     )
                                
                            #     # Von Neumann Entropy
                            #     singular_values_sq_sum = singular_values_sq.sum(dim=1)                          # (num_patches, )
                            #     fraction = singular_values_sq / (singular_values_sq_sum.unsqueeze(1) + 1e-12)   # (num_patches, min(B*P_u*P_v, D))
                            #     neuman_entropy = -torch.sum(fraction * torch.log(fraction + 1e-9), dim=1)       # (num_patches, )
                            #     h5f_data_group["optimal_neumann_entropies"][start_t, :] = neuman_entropy.cpu().numpy()
                            #     del fraction, neuman_entropy
                                
                            #     # Stable Rank
                            #     # cf.) Frobenious Norm Squared is equivalent to the Square Sum of Singular Values
                            #     spectral_norm_sq, _ = torch.max(singular_values_sq, dim=1)                # (num_pathces, )
                            #     stable_rank = singular_values_sq_sum / (spectral_norm_sq + 1e-12)
                            #     h5f_data_group["optimal_stable_ranks"][start_t, :] = stable_rank.cpu().numpy()
                            #     del singular_values_sq, singular_values_sq_sum, spectral_norm_sq, stable_rank
                            # del batch_token_centered
                                
                            # Fourier Entropy        
                            fourier_coeffs_d2 = torch.einsum('vupd,fv,Fu->fFpd', mean_var_activation, fourier_basis_d1, fourier_basis_d1)
                            fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, num_patches, hidden_dim)                                    
                            powers = fourier_coeffs_d2.pow(2)                               # ((1+2*num_freqs)^2, num_patches, D)
                            powers = powers.sum(dim=2)                                      # ((1+2*num_freqs)^2, num_patches)
                            powers_by_freqs = powers[two_degree_idxes]                      # (num_freqs, num_2d_bases, num_patches)
                            powers_by_freqs = powers_by_freqs.sum(dim=1)                    # (num_freqs, num_patches)
                            powers_sum = powers_by_freqs.sum(dim=0) + 1e-9                  # (num_patches, )
                            fves = (powers_by_freqs / powers_sum).T                         # (num_freqs, num_patches)
                            # h5f_fves.create_dataset(
                            #     f"{start_t:02d}", data=fves.cpu().numpy(),
                            #     compression="gzip", compression_opts=4,
                            # )
                            
                            optimal_fves[start_t] = fves
                            print(f"\t Optimal FVE added")
                            # fourier_entropy = dist.Categorical(probs=fves).entropy()        # (num_patches, )
                            # h5f_data_group["optimal_fourier_entropies"][start_t, :] = fourier_entropy.cpu().numpy()     # (num_patches, ) added.
                            del fourier_coeffs_d2, powers, powers_by_freqs, powers_sum
                            del mean_activation                        
                        del batch_activation
                        
            # Calculate ODE Final Accuracy   
            logits = classifier(x_tgt.detach())
            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            correct_mask = (preds == answer_label)
            accuracy = correct_mask.float().mean()
            # h5f_data_group["ode_final_accuracies"][start_t] = accuracy.item()
            print(f"\t ODE Accuracy : {accuracy:.2f}")
            ode_accuracies[start_t] = accuracy
            
            # Save Final Image
            save_img_path = os.path.join(img_dir, f"label_{save_img_label}-start_t_{start_t}-final.png")
            img_tenser_cpu = x_tgt[save_img_idx].detach().cpu().clone()
            save_img_list.append((save_img_path, img_tenser_cpu))
                
            cache.clear()
            del cache
            gc.collect()
            torch.cuda.empty_cache()
        
        # ODE Failure timestep
        temp_ode_failure_tdx = -1
        temp_ode_failure_val = -1
        # for t_idx, t_accuarcy in enumerate(h5f_data_group["ode_final_accuracies"]):
        #     if temp_ode_failure_tdx == -1 and t_accuarcy <= ode_fail_threshold_val:
        #         temp_ode_failure_tdx = t_idx
        #         temp_ode_failure_val = t_accuarcy
        #         break
        # h5f_data_group.attrs["ode_fail_threshold_tdx"] = temp_ode_failure_tdx
        # h5f_data_group.attrs["ode_fail_threshold_val"] = temp_ode_failure_val
        
    print(f"Start saving sample images.")        
    for save_path, save_img in save_img_list:
        if "initial" in save_path:
            tvu.save_image(save_img, save_path, normalize=True, value_range=(-1, 1))
        else:
            tvu.save_image(save_img, save_path)
    del save_img_list
    print(f"fin.")   