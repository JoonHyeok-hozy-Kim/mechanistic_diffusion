import os
from pathlib import Path
from datetime import datetime as dt
import wandb
import torch

from utils.wandb import get_best_checkpoint, get_wandb_run_info, find_project_run_dirctory


def get_project_root_path(file_path, marker=".git"):
    curr_dir = Path(file_path).resolve()
    
    for parent in [curr_dir] + list(curr_dir.parents):
        if (parent / marker).exists():
            return parent
            
        if parent == parent.parent:
            break
            
    raise FileNotFoundError(f"Could not find the project directory with marker '{marker}'.")


def make_weight_and_result_dir(
    in_weight_dir, 
    in_result_dir, 
    dry_run, 
    resume_wandb_url, 
    settings_str, 
    wandb_exp_name, 
    exp_config_dict,
    prev_lr_schedule_type='ws',
):
    now_str = dt.now().strftime("%y%m%d_%H%M%S")    
    prev_weight_dir = None
    lr_schedule_type = exp_config_dict["lr_schedule_type"]
    
    if dry_run:
        base_weight_dir = os.path.join(in_weight_dir, f"dry_run-{now_str}")
        base_result_dir = os.path.join(in_result_dir, f"dry_run-{now_str}")
        
    else:
        if resume_wandb_url and resume_wandb_url.lower() != "false":
            user_name, project_name, run_id = get_wandb_run_info(resume_wandb_url)
            print(f"user_name, project_name, run_id : {user_name}, {project_name}, {run_id}")
            
            wandb.init(
                project=project_name,
                entity=user_name, 
                id=run_id,
                resume="must",
            )
            
            base_weight_dir = find_project_run_dirctory(in_weight_dir, run_id, 2)
            base_result_dir = find_project_run_dirctory(in_result_dir, run_id, 2)
            print(f"found base_weight_dir : {base_weight_dir}")
            print(f"found base_result_dir : {base_result_dir}")
            print(f"Resume training.")
            
            lr_schedule_type_dir = 'lr_schedule_type' + '_' + prev_lr_schedule_type
            prev_weight_dir = os.path.join(base_weight_dir, lr_schedule_type_dir)
            assert os.path.exists(prev_weight_dir), f"weights_dir {prev_weight_dir} does not exist for resuming."
            
                
        else:
            run_id = os.environ.get("WANDB_RUN_ID", wandb.util.generate_id())
            print(f"Initiate wandb with new run_id : {run_id}")
            
            wandb.init(
                project=wandb_exp_name, 
                name=settings_str,
                id=run_id,
                resume="allow",
                config = exp_config_dict,
            )
                
            base_weight_dir = os.path.join(in_weight_dir, f"wandb-{now_str}-{run_id}-{settings_str}")
            base_result_dir = os.path.join(in_result_dir, f"wandb-{now_str}-{run_id}-{settings_str}")
    
    # Directory settings based on wandb run_id
    new_weights_dir = os.path.join(base_weight_dir, f"lr_schedule_type_{lr_schedule_type}")
    new_results_dir = os.path.join(base_result_dir, f"lr_schedule_type_{lr_schedule_type}")    
    os.makedirs(new_weights_dir, exist_ok=True)
    os.makedirs(new_results_dir, exist_ok=True)
    
    return new_weights_dir, new_results_dir, prev_weight_dir
                
                
                
def get_checkpoint_and_resume(prev_weights_dir, optimizer, model):
    print(f"=> Getting checkpoints from: {prev_weights_dir}")
    last_checkpoint = get_best_checkpoint(prev_weights_dir)
    checkpoint = torch.load(last_checkpoint, weights_only=False)
    start_epoch = checkpoint['epoch'] + 1
    
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"=> Loaded model weights.")
    
    if 'min_train_loss' in checkpoint: 
        min_train_loss = checkpoint['min_train_loss']
    
    # if 'min_valid_loss' in checkpoint: 
    #     min_valid_loss = checkpoint['min_valid_loss']
    
    print(f"=> Resuming from epoch {start_epoch}")
    
    
def get_config_str_dict(args):
    config_dict = {}
    config_str_list = []
    if "dry_run" in args and args.dry_run:
        config_str_list.append("dry_run")
    print(f'{" Config ":-^80}')
    for arg_key, arg_val in vars(args).items():
        print(f'{arg_key:32s}: {arg_val}')
        config_dict[arg_key] = arg_val
                
        if arg_key in (
            "generative_model_path", "classifier_weights_path", "output_dir", 
            "epochs", "batch_size", "sample_freq", "dry_run", "resume_wandb_url",
            "source_data_confidence_threshold", "prev_lr_schedule_type",
            "channel_size", "image_size",
        ):
            continue
        elif type(arg_val) == bool:
            continue
            # config_str_list.append(f"{arg_key}")
        else:
            config_str_list.append(f"{arg_key}_{arg_val}")    
    
    config_str = "-".join(config_str_list)
    return config_dict, config_str

                
                
def get_checkpoint_start_epoch_and_resume(prev_weights_dir, optimizer, model, choose_final=False):
    print(f"=> Getting checkpoints from: {prev_weights_dir}")
    last_checkpoint = get_best_checkpoint(prev_weights_dir, choose_final)
    checkpoint = torch.load(last_checkpoint, weights_only=False)
    start_epoch = checkpoint['epoch'] + 1
    
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"=> Loaded model weights.")
    
    if 'min_train_loss' in checkpoint: 
        min_train_loss = checkpoint['min_train_loss']
    
    # if 'min_valid_loss' in checkpoint: 
    #     min_valid_loss = checkpoint['min_valid_loss']
    
    print(f"=> Resuming from epoch {start_epoch}")
    
    return start_epoch




def make_result_dir(in_result_dir, settings_str):
    now_str = dt.now().strftime("%y%m%d_%H%M%S")    
    ret_result_dir = os.path.join(in_result_dir, f"{now_str}-{settings_str}")
    os.makedirs(ret_result_dir, exist_ok=True)
    
    return ret_result_dir