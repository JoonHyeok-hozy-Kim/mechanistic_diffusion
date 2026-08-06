import argparse
import os
from datetime import datetime as dt

import torch
from torch.utils.data import DataLoader

from utils.manage_project import get_config_str_dict, make_result_dir
from utils.seed import seed_everything
from models.dit import DiT_Llama
from models.rf import RF
from data.manage_dataset import get_confident_image_data_dict
from data.modular_addition import Operand_Sorted_Modular_Arithmetic_Dataset
from classifier_models.manage_classifiers import get_classifier
from analyses.utils.fourier_analysis import get_fourier_basis_and_names, get_indices_all_frequencies, \
    get_d2_indices_by_frequency
from analyses.utils.plot import save_patch_grid_plot_single_variable

RANDOM_SEED = 46


def generate_projection_fve_table(p, activation, dataset, save_dir, latex=False):
    assert len(activation.shape) == 3, f"Unexpected input activation shape of {activation.shape}. (Expected a single patch (p, p, hidden_dim))"
    
    device = activation.device
    hidden_dim = activation.shape[2]
        
    fourier_basis_d1, _ = get_fourier_basis_and_names(p, latex=latex)
    num_freqs = (p - 1) // 2
    
    c_extracted = torch.clone(activation).to(device)
    for idx in range(len(dataset)):
        a, b, c = dataset.get_num_data(idx)
        c_extracted[c, a, :] = activation.reshape(p**2, hidden_dim)[idx, :]
    
    uv_ks = torch.einsum('cah, kc -> kh', c_extracted, fourier_basis_d1)    # (1+2*num_freqs, hidden)
    uv_ks = uv_ks / p   # Normalize the P_a values added by einsum.
    uv_ks_mlp = torch.einsum('kh, abh -> kab', uv_ks, activation)           # (1+2*num_freqs, P_a, P_b)
    # uv_ks_mlp = uv_ks_mlp.reshape(1+2*num_freqs, p**2)                      # (1+2*num_freqs, P_a*P_b)
    
    f = open(save_dir, "w", encoding="utf-8")
    # f.write(f"W_L Component,Fourier Components on u_k^T MLP(a,b) and v_k^T MLP(a,b),FVE\n")
    if latex:
        f.write(f"|$W_L$|$u_k^\\top \\text{{MLP}}(a,b) \\text{{ and }} v_k^\\top \\text{{MLP}}(a,b)$|FVE|\n")
    else:
        f.write(f"|W_L|u_k^T MLP(a,b) and v_k^T MLP(a,b)|FVE|\n")
    f.write(f"|:-:|:-:|:-:|\n")
        
    for k in range(1, num_freqs+1):
        frequencies_dict = {}
        frequencies_dict[2*k - 1] = f'$\\cos(w_{{{k}}})$' if latex else f'cos(w_{k})'
        frequencies_dict[2*k] = f'$\\sin(w_{{{k}}})$' if latex else f'sin(w_{k})'
        
        target_indices_and_desc = get_d2_indices_by_frequency(args.p, k, latex=latex)
        target_indices = [idx for idx, desc in target_indices_and_desc]
        target_indices_desc = [desc for idx, desc in target_indices_and_desc]
        
        for curr_freq_idx in frequencies_dict.keys():
            f.write(f"|{frequencies_dict[curr_freq_idx]}|")
            
            curr_uv_ks_mlp = uv_ks_mlp[curr_freq_idx, :].unsqueeze(-1)    # (P_a, P_b, 1)
            
            d2_result = torch.einsum('abx, fa, Fb -> fFx', curr_uv_ks_mlp, fourier_basis_d1, fourier_basis_d1)
            d2_result = d2_result.squeeze().reshape(args.p**2)          # ((1+2*num_freqs)^2,)
            d2_result_pow = d2_result.pow(2)                            # ((1+2*num_freqs)^2,)
            d2_fves = d2_result_pow / (d2_result_pow[1:].sum() + 1e-9)  # ((1+2*num_freqs)^2,) : Exclude Constant term for FVE calculation
            
            curr_freq_coeffs = d2_result[target_indices]
            two_d_fves_targets = d2_fves[target_indices]
            
            col_proj_txt_list = []            
            if 'cos' in frequencies_dict[curr_freq_idx].split("(")[0]:
                col_proj_txt_list.append(f"{curr_freq_coeffs[4]:,.2f}{target_indices_desc[4]}")  # cos*cos
                if curr_freq_coeffs[7] >= 0:
                    col_proj_txt_list.append("+")
                col_proj_txt_list.append(f"{curr_freq_coeffs[7]:,.2f}{target_indices_desc[7]}")  # sin*sin
                col_fve_txt = f"{(two_d_fves_targets[4]+two_d_fves_targets[7]):.2f}"
            else:
                col_proj_txt_list.append(f"{curr_freq_coeffs[5]:,.2f}{target_indices_desc[5]}")  # cos*sin
                if curr_freq_coeffs[6] >= 0:
                    col_proj_txt_list.append("+")
                col_proj_txt_list.append(f"{curr_freq_coeffs[6]:,.2f}{target_indices_desc[6]}")  # sin*cos
                col_fve_txt = f"{(two_d_fves_targets[5]+two_d_fves_targets[6]):.2f}"
            
            if latex:
                col_proj_txt_list.insert(0, "$")
                col_proj_txt_list.append("$")
            
            col_proj_txt = ''.join(col_proj_txt_list)
            f.write(col_proj_txt)
            f.write("|")
            f.write(col_fve_txt)
            f.write("|\n")
            
            # if 'cos' in frequencies[curr_freq_idx].split("(")[0]:
            #     f.write(f"{curr_freq_coeffs[4]:.2f}{target_indices_desc[4]}")   # cos*cos
            #     if curr_freq_coeffs[7] >= 0:
            #         f.write("+")
            #     f.write(f"{curr_freq_coeffs[7]:.2f}{target_indices_desc[7]}")   # sin*sin
            #     f.write("|")                
            #     f.write(f"{(two_d_fves_targets[4]+two_d_fves_targets[7]):.2f}")
            #     f.write("|\n")
                
            # else:
            #     f.write(f"{curr_freq_coeffs[5]:.2f}{target_indices_desc[5]}")   # cos*sin
            #     if curr_freq_coeffs[6] >= 0:
            #         f.write("+")
            #     f.write(f"{curr_freq_coeffs[6]:.2f}{target_indices_desc[6]}")   # sin*cos
            #     f.write("|")                
            #     f.write(f"{(two_d_fves_targets[5]+two_d_fves_targets[6]):.2f}")
            #     f.write("|\n")                
    f.close()


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--generative_model_path", type=str, default="weights/")
        parser.add_argument("--classifier_weights_path", type=str, default="weights/classifier_resnet_reserved")
        parser.add_argument("--output_dir", type=str, default="results/")
        parser.add_argument("--dataset_name", type=str, default="emnist", help="Dataset type. e.g. emnist, kmnist, two_dimensional")
        parser.add_argument("--p", type=int, default=97, help="Modulus for modular arithmetic (should be <= 97 for all characters: A-Z, a-z, 0-9, special chars)")
        # parser.add_argument("--train_fraction", type=float, default=0.7)
        parser.add_argument("--num_images_per_class", type=int, default=1)
        parser.add_argument("--source_data_confidence_threshold", type=float, default=0.9, help="Confidence threshold for adding source image to the dataset")       
        parser.add_argument("--patch_size", type=int, default=2, help="Patch size for the model")
        parser.add_argument("--model_size", type=str, default="S", help="S, M, L, XL")
        parser.add_argument("--prediction_target", type=str, choices=["x", "v", "epsilon"], help="DiT's prediction target provided to RF")
        parser.add_argument("--noise_schedule", type=str, default="linear", help="RF noise schedule")
        parser.add_argument("--depth", type=int, default=10)
        parser.add_argument("--ffn_type", type=str, default="gelu", choices=["swiglu", "gelu"], help="Choose FFN type : swiglu, gelu")
        parser.add_argument("--use_ffn_sandwich", action="store_true", help="Use prepending ffn to self-attention")
        # parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
        # parser.add_argument("--lr_schedule_type", type=str, default="wsd", choices=["wsd", "ws", "d", "cos"], help="Choose LR scheduler type")
        # parser.add_argument("--weight_decay", type=float, default=10.0, help="Weight decay for optimizer")
        parser.add_argument("--use_layer_norm", action="store_true", help="Apply LayerNorm in Transformer and FinalLayer blocks")
        parser.add_argument("--use_rope_2d", action="store_true", help="Apply rope 2d on Attention layers")
        parser.add_argument("--use_ffn_residual", action="store_true", help="Use residual connection at FFN")
        # parser.add_argument("--epochs", type=int, default=1)
        # parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--sample_steps", type=int, default=1)
        # parser.add_argument("--sample_freq", type=int, default=1)
        # parser.add_argument("--dry_run", action="store_true", help="Do not recored to wandb")
        # parser.add_argument("--resume_wandb_url", type=str, default=None, help="URL of the run to resume")
        parser.add_argument("--correct_result_img", action="store_true", help="Correct result image at position c")
        parser.add_argument("--latex_formatted_out", action="store_true", help="Latex formatted output")
        parser.add_argument("--ffn_out_pre_act_only", action="store_true", help="Target only the FFN out Pre-activation layer.")
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
    
    ds_all = Operand_Sorted_Modular_Arithmetic_Dataset(
        p=args.p, split='all', num_images_per_class=args.num_images_per_class, img_data_dict=source_img_data_dict
    )    
    len_data = len(ds_all.data)         
    dl_all = DataLoader(ds_all, batch_size=len_data, shuffle=False, drop_last=False, num_workers=0)    
    
    # Inference and capture all the activations
    images = []
    x_tgt_base = torch.randn(1, 1, 32, 32).to(device)
    for batch_idx, (_, x_src, label_tgt, _, _) in enumerate(dl_all):
        x_src = x_src.to(device)
        # x_tgt = torch.randn(x_src.size(0), 1, 32, 32).to(device)
        x_tgt = x_tgt_base.repeat(x_src.size(0), 1, 1, 1)
        images.append(rf.sample(x_tgt, x_src, sample_steps=args.sample_steps))
        
    # Foureir Analysis Settings
    fourier_basis_d1, fourier_basis_names = get_fourier_basis_and_names(args.p)  
    variables = ['c', 'a', 'b']
    num_variables = len(variables)
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
    
    # Get idices of 1-degree bases and 2-degree bases with same frequences
    two_degree_idxes, two_degree_idxes_descs = get_indices_all_frequencies(args.p)  # (num_ks, num_2d_bases)  e.g. (11, 8) 
    indices_flattened = two_degree_idxes.flatten()                                  # (num_ks * num_2d_bases)
    num_freqs = (args.p - 1) // 2
    num_2d_bases = num_variables**2 - 1         # Exclude the constant basis : {cos(a), sin(a), cos(b), sin(b), cos(a)*cos(b), ... , sin(a)*sin(b)}
    
    fve_plots_dir = os.path.join(result_dir, 'fve_plots')
    fve_uv_k_proj_dir = os.path.join(result_dir, 'fve_uv_k_proj')
    
    for ldx, layer_name in enumerate(cache.keys()):
        # Exclude moments hooks
        if isinstance(cache[layer_name], dict):
            # print(f"Pass {layer_name}")
            continue
        
        print(f"L{ldx:02d}={layer_name} ", end="")
        layer_activation = cache[layer_name]            
        layer_dir = os.path.join(fve_plots_dir, f"L{ldx:02d}-{layer_name}")
        
        # After Embedding
        if layer_name in (
                'layers.0.hook_attn_norm', 'layers.0.hook_attn_adaLN'   # (B, S, D)
            ):               
            if args.ffn_out_pre_act_only:
                print("==> skip")
                continue
            os.makedirs(layer_dir, exist_ok=True)
            
            hidden_dim = layer_activation.shape[2]
            layer_activation = layer_activation.view(args.p, args.p, num_patches * num_variables, hidden_dim)  # (P_a, P_b, num_patches * num_variables, D)
            
            for vdx, variable in enumerate(variables):   # c, a, b
                var_d2_activation = layer_activation[:, :, patch_indices_by_variable[vdx], :]   # (P_a, P_b, num_patches, D)     
                if variable == 'c':
                    # Reorder (P_a, P_b, ...) -> (P_c, P_a, ...)
                    temp_var_d2_activation = var_d2_activation.clone()
                    for idx in range(len(ds_all)):
                        a, b, c = ds_all.get_num_data(idx)
                        # print(f"({a}, {b}) -> ({c}, {a})")
                        temp_var_d2_activation[c, a, :] = var_d2_activation[a, b]
                    var_d2_activation = temp_var_d2_activation          # (P_c, P_a, num_patches, D)
                    var_d1_activation = var_d2_activation.mean(dim=1)   # (P_c, num_patches, D)
                else:
                    vdx_data = vdx - 1
                    udx_data = (vdx_data + 1) % 2
                    var_d1_activation = var_d2_activation.mean(dim=udx_data) # (P_v, num_patches, D)
                    var_d1_activation_0 = torch.select(var_d2_activation, dim=udx_data, index=0)
                    assert torch.allclose(var_d1_activation, var_d1_activation_0, atol=1e-4), f"Check Degree-1 Activation indexation"
                
                foureir_d1_coeffs_patches = []
                foureir_d2_coeffs_patches = []
                
                for pdx in range(num_patches):
                    patch_d2_activation = var_d2_activation[:, :, pdx, :]   # (P_v, P_u, D)
                    patch_d1_activation = var_d1_activation[:, pdx, :]      # (P_v, D)
                    
                    # Degree one projection
                    fourier_coeffs_d1 = torch.einsum('vd, kv -> kd', patch_d1_activation, fourier_basis_d1)    # (1+2*num_freqs, D)
                    foureir_d1_coeffs_patches.append(fourier_coeffs_d1)
                    
                    # Degree two projection
                    fourier_coeffs_d2 = torch.einsum('vud,fv,Fu->fFd', patch_d2_activation, fourier_basis_d1, fourier_basis_d1)
                    fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, hidden_dim)
                    foureir_d2_coeffs_patches.append(fourier_coeffs_d2)
                
                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='bar_two_degree_neuron_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )
                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='heatmap_neuron_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )
        
        # Attention Q,K,V for individual operands 
        elif (  
                (layer_name.split("_")[-2] == "by" and layer_name.split("_")[-1] == 'head') or \
                (layer_name.split("_")[1] == 'rope')
            ):                                                  # (B, num_patches * num_variables, num_heads, head_dim)
            if args.ffn_out_pre_act_only:
                print("==> skip")
                continue
            os.makedirs(layer_dir, exist_ok=True)
        
            num_heads, head_dim = layer_activation.shape[2:]  # num_heads, head_dim
            
            layer_activation = layer_activation.reshape(args.p, args.p, num_patches * num_variables, num_heads, head_dim)    # (P_a, P_b, num_patches * num_variables, num_heads, head_dim)
            
            for vdx, variable in enumerate(variables):   # c, a, b
                var_d2_activation = layer_activation[:, :, patch_indices_by_variable[vdx], :, :]   # (P_a, P_b, num_patches, num_heads, head_dim)     
                if variable == 'c':
                    # Reorder (P_a, P_b, ...) -> (P_c, P_a, ...)
                    temp_var_de_activation = var_d2_activation.clone()
                    for idx in range(len(ds_all)):
                        a, b, c = ds_all.get_num_data(idx)
                        # print(f"({a}, {b}) -> ({c}, {a})")
                        temp_var_de_activation[c, a, :] = var_d2_activation[a, b]
                    var_d2_activation = temp_var_de_activation          # (P_c, P_a, num_patches, num_heads, head_dim)
                    var_d1_activation = var_d2_activation.mean(dim=1)   # (P_c, num_patches, num_heads, head_dim)
                else:
                    vdx_data = vdx - 1
                    udx_data = (vdx_data + 1) % 2
                    var_d1_activation = var_d2_activation.mean(dim=udx_data) # (P_v, num_patches, num_heads, head_dim)
                    var_d1_activation_0 = torch.select(var_d2_activation, dim=udx_data, index=0)
                    assert torch.allclose(var_d1_activation, var_d1_activation_0, atol=1e-4), f"Check Degree-1 Activation indexation"
                
                foureir_d1_coeffs_patches = []
                foureir_d2_coeffs_patches = []                
                
                for pdx in range(num_patches):
                    patch_d2_activation = var_d2_activation[:, :, pdx, :, :]   # (P_v, P_u, num_heads, head_dim)
                    patch_d1_activation = var_d1_activation[:, pdx, :, :]      # (P_v, num_heads, head_dim)
                    
                    # Degree one projection
                    fourier_coeffs_d1 = torch.einsum('vhd, kv -> khd', patch_d1_activation, fourier_basis_d1)    # (1+2*num_freqs, num_heads, head_dim)
                    
                    # Degree two projection
                    fourier_coeffs_d2 = torch.einsum('vuhd,fv,Fu->fFhd', patch_d2_activation, fourier_basis_d1, fourier_basis_d1)
                    fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, num_heads, head_dim)
                    foureir_d2_coeffs_patches.append(fourier_coeffs_d2)
                
                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='bar_two_degree_head_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )                    

                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='heatmap_head_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )
            
        elif layer_name in (        # (P^2, H, S_query, S_key)
                'layers.0.attention.hook_A_raw', 'layers.0.attention.hook_A_softmax'
            ):          
            if args.ffn_out_pre_act_only:
                print("==> skip")
                continue
            os.makedirs(layer_dir, exist_ok=True)
            
            num_heads = layer_activation.shape[1]
            cdx = 0  # 'c' variable index in variables
            
            for vdx, variable in enumerate(variables):   # c, a, b 
                if variable == 'c':
                    assert cdx == vdx, f"cdx {cdx} != vdx {vdx}"
                    continue    # Attention score not available for 'c'
                
                var_d2_activation = layer_activation[:, :, patch_indices_by_variable[cdx], :]       # (P_v*P_u,  num_heads, query_patches, S_key)
                var_d2_activation = var_d2_activation[:, :, :, patch_indices_by_variable[vdx]]      # (P_v*P_u,  num_heads, query_patches, key_patches)
                var_d2_activation = var_d2_activation.reshape(args.p, args.p, num_heads, num_patches, num_patches)  # (P_v, P_u, num_heads, query_patches, key_patches)
                                
                vdx_data = vdx - 1
                udx_data = (vdx_data + 1) % 2
                var_d1_activation = var_d2_activation.mean(dim=udx_data) # (P_v, num_heads, query_patches, key_patches)
                
                # No validation for the softmax case : P_u starts to affect P_v (Softmax a = exp{a}/sum(exp{a}))
                if layer_name != 'layers.0.attention.hook_A_softmax':   
                    var_d1_activation_0 = torch.select(var_d2_activation, dim=udx_data, index=0)
                    assert torch.allclose(var_d1_activation, var_d1_activation_0, atol=1e-4), f"Check Degree-1 Activation indexation"
                
                # # Centering on raw attention score?
                # if layer_name == 'layers.0.attention.hook_A_raw':
                #     var_d2_activation -= torch.mean(var_d2_activation, dim=udx_data, keepdim=True)
                    
                foureir_d2_coeffs_patches = [[] for _ in range(num_patches)]
                
                for k_pdx in range(num_patches):
                    k_patch_d2_activation = var_d2_activation[:, :, :, :, k_pdx]  # (P_v, P_u, num_heads)
                    k_patch_d1_activation = var_d1_activation[:, :, :, k_pdx]     # (P_v, num_heads)
                    for q_pdx in range(num_patches):
                        q_patch_d2_activation = k_patch_d2_activation[:, :, :, q_pdx]  # (P_v, P_u, num_heads)
                        q_patch_d1_activation = k_patch_d1_activation[:, :, q_pdx]
            
                        # Degree One Projection
                        fourier_coeffs_d1 = torch.einsum('vh, kv -> kh', q_patch_d1_activation, fourier_basis_d1)   # (1+2*num_freqs, num_heads)    
                        fourier_coeffs_d1 = fourier_coeffs_d1.unsqueeze(-1)
                                    
                        # Degree Two Projection
                        fourier_coeffs_d2 = torch.einsum('vuh,fv,Fu->fFh', q_patch_d2_activation, fourier_basis_d1, fourier_basis_d1) # (1+2*num_freqs, 1+2*num_freqs, num_heads)
                        fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, num_heads).unsqueeze(-1)                    # ((1+2*num_freqs)^2, num_heads, 1)
                        foureir_d2_coeffs_patches[k_pdx].append(fourier_coeffs_d2)
                
                for k_pdx, fourier_d2_list in enumerate(foureir_d2_coeffs_patches):                    
                    save_patch_grid_plot_single_variable(
                        grid_size, fourier_d2_list, num_freqs, layer_dir, var_name=variables[vdx], plot_type='bar_two_degree_head_level',
                        num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                    )                         
                    
                    save_patch_grid_plot_single_variable(
                        grid_size, fourier_d2_list, num_freqs, layer_dir, var_name=variables[vdx], plot_type='heatmap_head_level',
                        num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs,
                        k_pdx=k_pdx
                    )
            
                    
        elif (layer_name == 'layers.0.attention.hook_Av'): # (P^2, num_heads, num_variables*num_patches, head_dim) 
            if args.ffn_out_pre_act_only:
                print("==> skip")
                continue
            os.makedirs(layer_dir, exist_ok=True)
            
            num_heads = model.layers[0].attention.n_heads
            head_dim = model.layers[0].attention.head_dim        

            for vdx, variable in enumerate(variables):   # c, a, b
                var_d2_activation = layer_activation[:, :, patch_indices_by_variable[vdx], :]   # (P_a*P_b, num_heads, num_patches, head_dim)
                var_d2_activation = var_d2_activation.reshape(args.p, args.p, num_heads, num_patches, head_dim)
                if variable == 'c':
                    # Reorder (P_a, P_b, ...) -> (P_c, P_a, ...)
                    temp_var_de_activation = var_d2_activation.clone()
                    for idx in range(len(ds_all)):
                        a, b, c = ds_all.get_num_data(idx)
                        # print(f"({a}, {b}) -> ({c}, {a})")
                        temp_var_de_activation[c, a, :] = var_d2_activation[a, b]
                    var_d2_activation = temp_var_de_activation          # (P_c, P_a, num_heads, num_patches, head_dim)
                    var_d1_activation = var_d2_activation.mean(dim=1)   # (P_c, num_heads, num_patches, head_dim)
                else:
                    vdx_data = vdx - 1
                    udx_data = (vdx_data + 1) % 2
                    var_d1_activation = var_d2_activation.mean(dim=udx_data) # (P_v, num_heads, num_patches, head_dim)
                
                foureir_d1_coeffs_patches = []
                foureir_d2_coeffs_patches = []             
                
                for pdx in range(num_patches):
                    patch_d2_activation = var_d2_activation[:, :, :, pdx, :]   # (P_v, P_u, num_heads, head_dim)
                    patch_d1_activation = var_d1_activation[:, :, pdx, :]      # (P_v, num_heads, head_dim)
                    
                    # Degree one projection
                    fourier_coeffs_d1 = torch.einsum('vhd, kv -> khd', patch_d1_activation, fourier_basis_d1)    # (1+2*num_freqs, num_heads, head_dim)
                    foureir_d1_coeffs_patches.append(fourier_coeffs_d1)                    
                    
                    # Degree two projection
                    fourier_coeffs_d2 = torch.einsum('vuhd,fv,Fu->fFhd', patch_d2_activation, fourier_basis_d1, fourier_basis_d1)
                    fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, num_heads, head_dim)
                    foureir_d2_coeffs_patches.append(fourier_coeffs_d2)
                    
                    # draw_two_degree_heatmap(
                    #     fourier_coeffs_d2, num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs,
                    #     layer_dir, num_heads=num_heads, head_dim=head_dim,
                    #     file_name=f"FVE_2d_heatmap-variable_{variable}_patch_{pdx}", title=f"Variable {variable} - Patch {pdx}"
                    # )                    
                             
                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='bar_two_degree_head_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )             

                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='heatmap_head_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )

            
        else: 
            if args.ffn_out_pre_act_only and layer_name != 'layers.0.feed_forward_out.hook_pre_activation':
                print("==> skip")
                continue
            os.makedirs(layer_dir, exist_ok=True)
            
            curr_dim = layer_activation.shape[2]
            layer_activation = layer_activation.view(args.p, args.p, num_patches * num_variables, curr_dim)  # (P_a, P_b, num_patches * num_variables, D)
            
            for vdx, variable in enumerate(variables):   # c, a, b
                var_d2_activation = layer_activation[:, :, patch_indices_by_variable[vdx], :]   # (P_a, P_b, num_patches, D)     
                var_d1_activation = var_d2_activation.mean(dim=1)   # (P_c, num_patches, D)

                if variable == 'b':
                    vdx_data = vdx - 1
                    udx_data = (vdx_data + 1) % 2
                    var_d1_activation = var_d2_activation.mean(dim=udx_data) # (P_v, num_patches, D)
                    # var_d1_activation_0 = torch.select(var_d2_activation, dim=udx_data, index=0)
                    # assert torch.allclose(var_d1_activation, var_d1_activation_0, atol=1e-4), f"Check Degree-1 Activation indexation"
                
                foureir_d1_coeffs_patches = []
                foureir_d2_coeffs_patches = []
                
                for pdx in range(num_patches):
                    patch_d2_activation = var_d2_activation[:, :, pdx, :]   # (P_v, P_u, D)
                    patch_d1_activation = var_d1_activation[:, pdx, :]      # (P_v, D)
                    
                    # Generate projection FVE table
                    if layer_name == 'layers.0.feed_forward_out.hook_pre_activation' and \
                        variable == 'c':
                        table_dir = os.path.join(fve_uv_k_proj_dir, layer_name)
                        os.makedirs(table_dir, exist_ok=True)
                        table_file_name = os.path.join(table_dir, f'var_{variable}-patch_{pdx}.md')
                        generate_projection_fve_table(args.p, patch_d2_activation, ds_all, table_file_name, args.latex_formatted_out)
                    
                    
                    # Degree one projection
                    fourier_coeffs_d1 = torch.einsum('vd, kv -> kd', patch_d1_activation, fourier_basis_d1)    # (1+2*num_freqs, D)
                    foureir_d1_coeffs_patches.append(fourier_coeffs_d1)
                    
                    # Degree two projection
                    fourier_coeffs_d2 = torch.einsum('vud,fv,Fu->fFd', patch_d2_activation, fourier_basis_d1, fourier_basis_d1)
                    fourier_coeffs_d2 = fourier_coeffs_d2.reshape((1+2*num_freqs)**2, curr_dim)
                    foureir_d2_coeffs_patches.append(fourier_coeffs_d2)
                    
                
                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='bar_two_degree_neuron_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )
                save_patch_grid_plot_single_variable(
                    grid_size, foureir_d2_coeffs_patches, num_freqs, layer_dir, var_name=variables[vdx], plot_type='heatmap_neuron_level',
                    num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs
                )
        
        
        print(f"fin.")