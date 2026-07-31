import argparse
import os
from datetime import datetime as dt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.manage_project import get_config_str_dict, make_weight_and_result_dir, get_checkpoint_start_epoch_and_resume
from utils.seed import seed_everything
from utils.lr import get_lr_schedule, set_lr
from models.dit import DiT_Llama
from models.rf import RF
from data.manage_dataset import get_image_data_dict
from data.modular_addition import Modular_Arithmetic_Dataset


RANDOM_SEED = 46

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--generative_model_path", type=str, default="weights/")
        parser.add_argument("--classifier_weights_path", type=str, default="weights/classifier_resnet_reserved")
        parser.add_argument("--output_dir", type=str, default="results/")
        parser.add_argument("--dataset_name", type=str, default="emnist", help="Dataset type. e.g. emnist, kmnist, two_dimensional")
        parser.add_argument("--p", type=int, default=97, help="Modulus for modular arithmetic (should be <= 97 for all characters: A-Z, a-z, 0-9, special chars)")
        parser.add_argument("--train_fraction", type=float, default=0.7)
        parser.add_argument("--num_images_per_class", type=int, default=1)
        parser.add_argument("--source_data_confidence_threshold", type=float, default=0.9, help="Confidence threshold for adding source image to the dataset")       
        parser.add_argument("--patch_size", type=int, default=2, help="Patch size for the model")
        parser.add_argument("--model_size", type=str, default="S", help="S, M, L, XL")
        parser.add_argument("--prediction_target", type=str, choices=["x", "v", "epsilon"], help="DiT's prediction target provided to RF")
        parser.add_argument("--noise_schedule", type=str, default="noise", help="RF noise schedule")
        parser.add_argument("--depth", type=int, default=10)
        parser.add_argument("--ffn_type", type=str, default="gelu", choices=["swiglu", "gelu"], help="Choose FFN type : swiglu, gelu")
        parser.add_argument("--use_ffn_sandwich", action="store_true", help="Use prepending ffn to self-attention")
        parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
        parser.add_argument("--lr_schedule_type", type=str, default="wsd", choices=["wsd", "ws", "d", "cos"], help="Choose LR scheduler type")
        parser.add_argument("--weight_decay", type=float, default=10.0, help="Weight decay for optimizer")
        parser.add_argument("--use_layer_norm", action="store_true", help="Apply LayerNorm in Transformer and FinalLayer blocks")
        parser.add_argument("--use_rope_2d", action="store_true", help="Apply rope 2d on Attention layers")
        parser.add_argument("--use_ffn_residual", action="store_true", help="Use residual connection at FFN")
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--sample_freq", type=int, default=1)
        parser.add_argument("--dry_run", action="store_true", help="Do not recored to wandb")
        parser.add_argument("--resume_wandb_url", type=str, default=None, help="URL of the run to resume")
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
    base_weight_dir = os.path.join(args.generative_model_path, script_dir_str)
    base_result_dir = os.path.join(args.output_dir, script_dir_str)
    
    weight_dir, result_dir, prev_weight_dir = make_weight_and_result_dir(
        base_weight_dir,
        base_result_dir,
        args.dry_run,
        args.resume_wandb_url,
        config_str,
        wandb_exp_name,
        config_dict,
    )
    
    
    
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
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")    
    
    rf = RF(model, time_sampling_type="uniform_skewed", noise_schedule=args.noise_schedule)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    # Resume from the previous run.
    if prev_weight_dir is not None:
        start_epoch = get_checkpoint_start_epoch_and_resume(prev_weight_dir, optimizer, model)
    else:
        start_epoch = 0
    
    # Needs Implementation!
    classifier = None
    source_data_confidence_threshold = 1.0
        
    source_img_data_dict = get_image_data_dict(
        args.dataset_name, args.p, args.num_images_per_class, 
        classifier, args.source_data_confidence_threshold,
    )
    
    
    ds_train = Modular_Arithmetic_Dataset(
        p=args.p, split='train', train_fraction=args.train_fraction,
        num_images_per_class=args.num_images_per_class, img_data_dict=source_img_data_dict
    )
    ds_valid = Modular_Arithmetic_Dataset(
        p=args.p, split='train', train_fraction=args.train_fraction,
        num_images_per_class=args.num_images_per_class, img_data_dict=source_img_data_dict
    )
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4)
    dl_valid = DataLoader(ds_valid, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("fin.")