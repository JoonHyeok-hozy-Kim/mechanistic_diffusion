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
from analyses.utils.plot import save_full_layers_grid

RANDOM_SEED = 46


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--generative_model_path", type=str, default="weights/")
        parser.add_argument("--classifier_weights_path", type=str, default="weights/classifier_resnet_reserved")
        parser.add_argument("--output_dir", type=str, default="results/")
        parser.add_argument("--dataset_name", type=str, default="emnist", help="Dataset type. e.g. emnist, kmnist, two_dimensional")
        parser.add_argument("--p", type=int, default=97, help="Modulus for modular arithmetic (should be <= 97 for all characters: A-Z, a-z, 0-9, special chars)")
        # parser.add_argument("--train_fraction", type=float, default=0.7)
        parser.add_argument("--num_images_per_class", type=int, default=256)
        parser.add_argument("--source_data_confidence_threshold", type=float, default=0.9, help="Confidence threshold for adding source image to the dataset")       
        parser.add_argument("--patch_size", type=int, default=2, help="Patch size for the model")
        parser.add_argument("--model_size", type=str, default="XL", help="S, M, L, XL")
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
        parser.add_argument("--sample_steps", type=int, default=50)
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
    
    ds_all = Inference_All_Image_Modular_Arithmetic_Dataset(
        p=args.p, num_images_per_class=args.num_images_per_class, 
        img_data_dict=source_img_data_dict, correct_result_img=args.correct_result_img
    )
    len_data = len(ds_all.data)         
    dl_all = DataLoader(ds_all, batch_size=len_data, shuffle=False, drop_last=False, num_workers=0)    
    
    variables = ['c', 'a', 'b']
    num_variables = len(variables)
    
    # Label for scatter plot
    lables_by_variable = [
        [(p_a + p_b)%args.p for p_a in range(args.p) for p_b in range(args.p)], # Label in 'c' order
        [p_a for p_a in range(args.p) for p_b in range(args.p)],    # Label in 'a' order
        [p_b for p_a in range(args.p) for p_b in range(args.p)],    # Label in 'b' order
    ]
    
    # Patch related parameters
    input_size = model.input_size
    patch_size = model.patch_size
    grid_size = input_size // patch_size
    num_patches = grid_size ** 2
            
    # Load Data : Ordered by (P_a, P_b) N^2 pairs in one batch
    dl_all_iter = iter(dl_all)
    answer_img, x_src, label_tgt, _, _ = next(dl_all_iter)
    
    # Sample from the operands dataset.
    x_src = x_src.to(device)
    answer_img = answer_img.to(device)
    x_tgt = torch.randn_like(answer_img, device=device)
    
    B = answer_img.size(0)
    dt = 1.0 / args.sample_steps
    dt = torch.tensor([dt] * B).to(device).view([B, *([1] * len(answer_img.shape[1:]))])
    time_step_cnt = 0
    
    with torch.no_grad():
        for i in range(args.sample_steps, 0, -1):
            print(f"Time Step Count : {i}")
            
            t_scalar = i / args.sample_steps
            if args.prediction_target == 'epsilon':
                t_scalar = min(t_scalar, 0.99)
            t_tensor = torch.tensor([t_scalar] * B).to(x_tgt.device)
            
            vc = model(x_tgt, t_tensor, x_src)
            x_tgt = x_tgt - dt * vc     # Instantiate new x_tgt everytime.
            
            curr_cache = {}
            for layer_name, activation in cache.items():
                if not isinstance(activation, torch.Tensor):
                    continue
                mean_activation = activation.detach().clone()
                prev_shape = mean_activation.shape
                mean_activation = mean_activation.reshape(ds_all.batch_repeat, (args.p)**2, *prev_shape[1:])
                mean_activation = torch.mean(mean_activation, dim=0)
                # print(f"{layer_name} : {mean_activation.shape}")
                curr_cache[layer_name] = mean_activation
                    
            # Time-wise Activations : variable - patch - layer
            curr_time_activations = defaultdict(lambda: defaultdict(dict))
            layer_cnt = 0
            for layer_name, activation in curr_cache.items():
                if not isinstance(activation, torch.Tensor):
                    continue
                print(f"{layer_cnt:2d}  {layer_name} : {activation.shape}")
                
                patch_indices_by_variable = []
                for i in range(num_variables):
                    indices = []
                    for j in range(grid_size):
                        var1 = i * grid_size + j * num_variables * grid_size
                        for k in range(grid_size):
                            var2 = var1 + k
                            indices.append(var2)
                    patch_indices_by_variable.append(indices)
                
                
                for vdx, variable in enumerate(variables):                    
                    var_activation = torch.clone(activation)                    
                    curr_var_patch_indices = patch_indices_by_variable[vdx]
                    print(f"\t[var {variable}] patch indices : {curr_var_patch_indices}")
                    
                    # Reshape Attention Score layers to start with (B, num_variables*num_patches, ...)
                    c_var_patch_indices = patch_indices_by_variable[0]
                    num_heads = model.layers[0].attention.n_heads
                    head_dim = model.layers[0].attention.head_dim
                    if 'hook_A_raw' in layer_name or 'hook_A_softmax' in layer_name:
                        if variable == 'c':    # No Attention Score for the variable c
                            continue
                        
                        # Original Shape : (P_u*P_v, num_heads, num_variables*num_patches, num_variables*num_patches)
                        var_activation = var_activation.reshape(
                            args.p**2, num_variables*num_patches, num_variables*num_patches, num_heads
                        )
                        # Slice query(c) patches
                        var_activation = var_activation[:, :, c_var_patch_indices, :]     # (P_u*P_v, num_variables*num_patches, c_patches, num_heads)
                    
                    elif 'hook_Av' in layer_name:
                        if variable == 'c':    # No Attention Score for the variable c
                            continue
                        
                        # Original Shape : (P_u*P_v, num_heads, num_variables*num_patches, head_dim)
                        var_activation = var_activation.reshape(
                            args.p**2, num_variables*num_patches, num_heads, head_dim
                        )
                    
                    var_activation = var_activation[:, curr_var_patch_indices]            # (P_u*P_v, u_patches, ...)
                    var_activation = var_activation.reshape(args.p**2, num_patches, -1)     # (P_u*P_v, u_patches, D)
                    print(f"\t\t[var_activation : {var_activation.shape}")                
                    
                    for pdx in range(num_patches):
                        patch_activation = var_activation[:, pdx]
                        
                        variable_dir = os.path.join(result_dir, f"variable_{variable}")
                        patch_dir = os.path.join(variable_dir, f"patch_{pdx}")
                        if time_step_cnt == 0:
                            os.makedirs(patch_dir, exist_ok=True)             
                        else:
                            assert os.path.exists(patch_dir), f"Dir does not exist : {patch_dir}"
                        
                        curr_time_activations[variable][pdx][layer_name] = patch_activation                    
                
                layer_cnt += 1
            
            for variable, val_pdx in curr_time_activations.items():         
                variable_labels = lables_by_variable[variables.index(variable)]
                variable_dir = os.path.join(result_dir, f"variable_{variable}") 
                for pdx, val_layer in val_pdx.items():
                    patch_dir = os.path.join(variable_dir, f"patch_{pdx}")
                    os.makedirs(patch_dir, exist_ok=True)  
                    out_path = os.path.join(patch_dir, f"all_layers_timestep_{time_step_cnt:02d}.png")
                    img_obj = save_full_layers_grid(val_layer, variable_labels, out_path, time_step_cnt, variable)
            
            curr_cache.clear()
            del curr_cache
            gc.collect()
            torch.cuda.empty_cache()
            time_step_cnt += 1
            
    for variable in variables:
        variable_dir = os.path.join(result_dir, f"variable_{variable}")
        if not os.path.exists(variable_dir): continue
        
        for pdx in range(num_patches):
            patch_dir = os.path.join(variable_dir, f"patch_{pdx}")
            if not os.path.exists(patch_dir): continue
            
            png_files = sorted([f for f in os.listdir(patch_dir) if f.endswith('.png')])
            if len(png_files) > 1:
                frames = []
                for filename in png_files:
                    img_path = os.path.join(patch_dir, filename)
                    img = Image.open(img_path)
                    frames.append(img)
                    
                gif_path = os.path.join(patch_dir, f"00_evolution_variable_{variable}_patch_{pdx}.gif")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,  # 200ms per frame (5 FPS)
                    loop=0         # repeat
                )
                print(f"✅ GIF Created: {gif_path}")