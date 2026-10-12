import argparse
import os
import math
import gc
from datetime import datetime as dt
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import DataLoader

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
        parser.add_argument("--channel_size", type=int, default=1, help="Channel size of the input image. 2 for 2D data.")
        parser.add_argument("--image_size", type=int, default=32, help="Width and Height of the input image")
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
        parser.add_argument("--latex_formatted_out", action="store_true", help="Latex formatted output")
        parser.add_argument("--plot_fve", action="store_true", help="Plot predictions and FVE barcharts.")
        parser.add_argument("--plot_pca", action="store_true", help="Plot PCA on every layer activations.")
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
        ffn_type=args.ffn_type, use_ffn_sandwich=args.use_ffn_sandwich, ffn_dim_multiplier=1,
        use_layer_norm=args.use_layer_norm, use_rope_2d=args.use_rope_2d, use_ffn_residual=args.use_ffn_residual, 
    )
    kwargs.update(common_kwargs)
            
    model = DiT_Llama(in_channels=args.channel_size, input_size=args.image_size, patch_size=args.patch_size, **kwargs).cuda()   
    
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
        
    timestep_sampling_type = "uniform_skewed" if args.dataset_name == 'two_dimensional' else 'normal'
    rf = RF(
        model, 
        timestep_sampling_type=timestep_sampling_type, 
        prediction_target=args.prediction_target, 
        noise_schedule=args.noise_schedule
    )
            
    classifier = get_classifier(dataset_name=args.dataset_name, num_classes=args.p).to(device)    
    source_img_data_dict = get_confident_image_data_dict(
        args.dataset_name, args.p, args.num_images_per_class, 
        classifier, args.source_data_confidence_threshold,
        img_height=args.image_size, img_width=args.image_size,
    )
    
    ds_all = Inference_All_Image_Modular_Arithmetic_Dataset(
        p=args.p, num_images_per_class=args.num_images_per_class, 
        img_data_dict=source_img_data_dict, correct_result_img=True,
    )
    len_data = len(ds_all.data)         
    dl_all = DataLoader(ds_all, batch_size=len_data, shuffle=False, drop_last=False, num_workers=0)    
    
    variables = ['c', 'a', 'b']
    num_variables = len(variables)
    
    # Patch related parameters
    grid_size = args.image_size // args.patch_size
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
    x_tgt = torch.randn_like(answer_img, device=device)
    
    # Fourier Analysis settings
    fourier_basis_d1, fourier_basis_names = get_fourier_basis_and_names(args.p, latex=args.latex_formatted_out)
    num_freqs = (args.p - 1) // 2
    num_2d_bases = num_variables**2 - 1    
    # Get idices of 1-degree bases and 2-degree bases with same frequences
    two_degree_idxes, two_degree_idxes_descs = get_indices_all_frequencies(args.p, latex=args.latex_formatted_out)  # (num_ks, num_2d_bases)  e.g. (11, 8) 
    indices_flattened = two_degree_idxes.flatten()  
    desc_flattened = two_degree_idxes_descs.flatten()
    fve_target_layers = {
        'layers.0.attention.hook_k': {
            'target_var': 'b',
            'store_idx': 0,
            'bar_chart_title': f"Operand 'b' Attention K Layer FVE values",
        },
        'layers.0.feed_forward_out.hook_pre_activation': {
            'target_var': 'c',
            'store_idx': 1,
            'bar_chart_title': f"Result 'c' FFN pre-GeLU Layer FVE values",
        }        
    }
    fve_all_time_max_val = [None, None]
    
    B = answer_img.size(0)
    dt = 1.0 / args.sample_steps
    dt = torch.tensor([dt] * B).to(device).view([B, *([1] * len(answer_img.shape[1:]))])
    time_step_cnt = 0
    
    # Save dirs
    fve_dir = os.path.join(result_dir, "plot_fve")
    pca_dir = os.path.join(result_dir, "plot_pca")
    
    with torch.no_grad():
        for i in range(args.sample_steps, 0, -1):
            print(f"Time Step Count : {i}")
            
            t_scalar = i / args.sample_steps
            if args.prediction_target == 'epsilon':
                t_scalar = min(t_scalar, 0.99)
            t_tensor = torch.tensor([t_scalar] * B).to(x_tgt.device)
            t_expanded = t_tensor.view([B, *([1] * len(x_tgt.shape[1:]))])
            
            # Sample
            pred = model(x_tgt, t_tensor, x_src)
            v_pred = rf.convert_pred_to_v_pred(x_tgt, pred, t_expanded)
            x_tgt = x_tgt - dt * v_pred     # Instantiate new x_tgt everytime.
            
            # Make predictions
            if args.plot_fve:            
                # Classification
                logits = classifier(x_tgt.detach()) 
                probs = torch.softmax(logits, dim=1)
                confidences, preds_tensor = torch.max(probs, dim=1)
                confidences = confidences.cpu().numpy()
                preds = preds_tensor.cpu().numpy()            
                accuracy = (preds == label_tgt).float().mean().item()
                correct_mask = (preds == label_tgt)   
                threshold = 0.8  
                low_confidence_mask = (confidences < threshold) # Confidence >= threshold masking
            
            # Record activations
            fve_activations = [[None, None] for _ in range(num_patches)]
            fve_bar_chart_titles = [None, None]
            pca_activations = defaultdict(lambda: defaultdict(dict))
            
            for layer_name, curr_activation in cache.items():
                if not isinstance(curr_activation, torch.Tensor):
                    continue                
                    
                if args.plot_fve and layer_name in fve_target_layers:
                    target_var = fve_target_layers[layer_name]['target_var']
                    vdx = variables.index(target_var)
                    store_idx = fve_target_layers[layer_name]['store_idx']
                    if fve_bar_chart_titles[store_idx] is None:
                        fve_bar_chart_titles[store_idx] = fve_target_layers[layer_name]['bar_chart_title']
                                    
                    hidden_dim = curr_activation.shape[2]
                    # batch_norm_activation = torch.clone(curr_activation)
                    batch_norm_activation = torch.clone(curr_activation).reshape(ds_all.batch_repeat, args.p, args.p, num_patches * num_variables, -1)
                    batch_norm_activation = torch.mean(batch_norm_activation, dim=0)    # (P, P, num_patches * num_tokens, D)
                    var_activation = batch_norm_activation[:, :, patch_indices_by_variable[vdx]] # (P, P, num_patches, D)
                    
                    for pdx in range(num_patches):
                        patch_activation = var_activation[:, :, pdx]
                        fourier_coeffs_d2 = torch.einsum('vud,fv,Fu->fFd', patch_activation, fourier_basis_d1, fourier_basis_d1)
                        fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, hidden_dim)
                        fve_activations[pdx][store_idx] = fourier_coeffs_d2
                            
                if args.plot_pca:                
                    for vdx, variable in enumerate(variables):             
                        var_activation = torch.clone(curr_activation)                    
                        curr_var_patch_indices = patch_indices_by_variable[vdx]
                        # print(f"\t[var {variable}] patch indices : {curr_var_patch_indices}")
                        
                        # Reshape Attention Score layers to start with (B, num_variables*num_patches, ...)
                        c_var_patch_indices = patch_indices_by_variable[0]
                        num_heads = model.layers[0].attention.n_heads
                        head_dim = model.layers[0].attention.head_dim
                        if 'hook_A_raw' in layer_name or 'hook_A_softmax' in layer_name:
                            if variable == 'c':    # No Attention Score for the variable c
                                continue
                            
                            # Original Shape : (P_u*P_v, num_heads, num_variables*num_patches, num_variables*num_patches)
                            var_activation = var_activation.reshape(
                                ds_all.batch_repeat * args.p**2, num_variables*num_patches, num_variables*num_patches, num_heads
                            )
                            # Slice query(c) patches
                            var_activation = var_activation[:, :, c_var_patch_indices, :]     # (P_u*P_v, num_variables*num_patches, c_patches, num_heads)
                        
                        elif 'hook_Av' in layer_name:
                            if variable == 'c':    # No Attention Score for the variable c
                                continue
                            
                            # Original Shape : (P_u*P_v, num_heads, num_variables*num_patches, head_dim)
                            var_activation = var_activation.reshape(
                                ds_all.batch_repeat * args.p**2, num_variables*num_patches, num_heads, head_dim
                            )
                        
                        var_activation = var_activation[:, curr_var_patch_indices]            # (P_u*P_v, u_patches, ...)
                        var_activation = var_activation.reshape(ds_all.batch_repeat * args.p**2, num_patches, -1)     # (P_u*P_v, u_patches, D)
                        # print(f"\t\t[var_activation : {var_activation.shape}")                
                    
                        for pdx in range(num_patches):
                            patch_activation = var_activation[:, pdx]                            
                            pca_activations[variable][pdx][layer_name] = patch_activation                    
                       
            # Plot FVE
            if args.plot_fve:
                for pdx in range(num_patches):
                    fve_patch_dir = os.path.join(fve_dir, f"patch_{pdx}")
                    os.makedirs(fve_patch_dir, exist_ok=True)  
                    fve_img_path = os.path.join(fve_patch_dir, f"fve_timestep_{time_step_cnt:02d}.png")
                    title = f"Inference and FVEs (Timestep: {time_step_cnt:02d}, Accy: {accuracy:.2f})"
                    curr_max_fve_val1, curr_max_fve_val2 = plot_fve_barchart_and_heatmap(
                        args.p, x_src, x_tgt, correct_mask, low_confidence_mask, 
                        fve_activations[pdx][0], fve_activations[pdx][1],
                        num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs,
                        fve_img_path, title=title, sub_bar1_title=fve_bar_chart_titles[0], sub_bar2_title=fve_bar_chart_titles[1],
                        max_fve_val1=fve_all_time_max_val[0], max_fve_val2=fve_all_time_max_val[1], 
                    )
                    fve_all_time_max_val[0] = curr_max_fve_val1 if fve_all_time_max_val[0] is None else max(curr_max_fve_val1, fve_all_time_max_val[0])
                    fve_all_time_max_val[1] = curr_max_fve_val2 if fve_all_time_max_val[1] is None else max(curr_max_fve_val2, fve_all_time_max_val[1])
            
            # Plot PCA                
            if args.plot_pca:             
                for variable, val_pdx in pca_activations.items():         
                    if variable == 'c':
                        variable_labels = label_tgt
                    elif variable == 'a':
                        variable_labels = label_a
                    elif variable == 'b':
                        variable_labels = label_b
                    else:
                        raise ValueError(f"Unexpected variable : {variable}")
                    
                    pca_var_dir = os.path.join(pca_dir, f"variable_{variable}") 
                    for pdx, val_layer in val_pdx.items():
                        pca_patch_dir = os.path.join(pca_var_dir, f"patch_{pdx}")                                                
                        os.makedirs(pca_patch_dir, exist_ok=True)  
                        pca_img_path = os.path.join(pca_patch_dir, f"all_layers_timestep_{time_step_cnt:02d}.png")
                        plot_pca_all_layers(val_layer, variable_labels, pca_img_path, time_step_cnt, variable, plot_centroid=True)
            
            gc.collect()
            torch.cuda.empty_cache()
            time_step_cnt += 1
    
    if args.plot_fve:
        for pdx in range(num_patches):
            patch_dir = os.path.join(fve_dir, f"patch_{pdx}")
            if not os.path.exists(patch_dir): continue
            fve_file_name = "00_animation.gif"
            generate_gif_animation(patch_dir, fve_file_name)

    if args.plot_pca:
        for variable in variables:
            variable_dir = os.path.join(pca_dir, f"variable_{variable}")
            if not os.path.exists(variable_dir): continue
            
            for pdx in range(num_patches):
                patch_dir = os.path.join(variable_dir, f"patch_{pdx}")
                if not os.path.exists(patch_dir): continue
                
                pca_file_name = "00_animation.gif"
                generate_gif_animation(patch_dir, pca_file_name)