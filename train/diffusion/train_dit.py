import argparse
import os
from datetime import datetime as dt
import wandb
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as tvu

from utils.manage_project import get_config_str_dict, make_weight_and_result_dir, get_checkpoint_start_epoch_and_resume
from utils.seed import seed_everything
from utils.lr import get_lr_schedule, set_lr
from models.dit import DiT_Llama
from models.rf import RF
from data.manage_dataset import get_confident_image_data_dict, save_sample_data
from data.modular_addition import Modular_Arithmetic_Dataset
from classifier_models.manage_classifiers import get_classifier


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
        parser.add_argument("--train_fraction", type=float, default=0.7)
        parser.add_argument("--num_images_per_class", type=int, default=1)
        parser.add_argument("--source_data_confidence_threshold", type=float, default=0.9, help="Confidence threshold for adding source image to the dataset")       
        parser.add_argument("--patch_size", type=int, default=2, help="Patch size for the model")
        parser.add_argument("--model_size", type=str, default="S", help="S, M, L, XL")
        parser.add_argument("--prediction_target", type=str, choices=["x", "v", "epsilon"], help="DiT's prediction target provided to RF")
        parser.add_argument("--noise_schedule", type=str, default="linear", help="RF noise schedule")
        parser.add_argument("--depth", type=int, default=10)
        parser.add_argument("--ffn_type", type=str, default="gelu", choices=["swiglu", "gelu"], help="Choose FFN type : swiglu, gelu")
        parser.add_argument("--use_ffn_sandwich", action="store_true", help="Use prepending ffn to self-attention")
        parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for optimizer")
        parser.add_argument("--lr_schedule_type", type=str, default="wsd", choices=["wsd", "ws", "d", "cos", "s"], help="Choose LR scheduler type")
        parser.add_argument("--weight_decay", type=float, default=10.0, help="Weight decay for optimizer")
        parser.add_argument("--use_layer_norm", action="store_true", help="Apply LayerNorm in Transformer and FinalLayer blocks")
        parser.add_argument("--use_rope_2d", action="store_true", help="Apply rope 2d on Attention layers")
        parser.add_argument("--use_ffn_residual", action="store_true", help="Use residual connection at FFN")
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--sample_freq", type=int, default=1)
        parser.add_argument("--dry_run", action="store_true", help="Do not recored to wandb")
        parser.add_argument("--resume_wandb_url", type=str, default=None, help="URL of the run to resume")
        parser.add_argument("--prev_lr_schedule_type", type=str, default="ws", help="Prev run lr_schedule_type. Usually either 'ws' or 's'.")
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
        args.prev_lr_schedule_type,
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
        
    model = DiT_Llama(in_channels=args.channel_size, input_size=args.image_size, patch_size=args.patch_size, **kwargs).cuda()
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")    
    
    rf = RF(model, time_sampling_type="uniform_skewed", noise_schedule=args.noise_schedule)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    # Resume from the previous run.
    if prev_weight_dir is not None:
        start_epoch = get_checkpoint_start_epoch_and_resume(prev_weight_dir, optimizer, model, choose_final=True)
    else:
        start_epoch = 0
    
    # Needs Implementation!
    classifier = get_classifier(dataset_name=args.dataset_name, num_classes=args.p).to(device)
    
    source_img_data_dict = get_confident_image_data_dict(
        args.dataset_name, args.p, args.num_images_per_class, 
        classifier, args.source_data_confidence_threshold,
        img_height=args.image_size, img_width=args.image_size,
    )
    
    ds_train = Modular_Arithmetic_Dataset(
        p=args.p, split='train', train_fraction=args.train_fraction,
        num_images_per_class=args.num_images_per_class, img_data_dict=source_img_data_dict
    )
    ds_valid = Modular_Arithmetic_Dataset(
        p=args.p, split='valid', train_fraction=args.train_fraction,
        num_images_per_class=args.num_images_per_class, img_data_dict=source_img_data_dict
    )    
    
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4)
    dl_valid = DataLoader(ds_valid, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
        
    save_weight_freq = args.sample_freq * 5
    curr_max_train_accuracy = -1
    curr_max_valid_accuracy = -1        
    tol = 0
    
    # Save model_best_records
    model_best_record = []
    model_best_record_file_path = os.path.join(weight_dir, f"model_best_record.txt")
    f = open(model_best_record_file_path, "w", encoding="utf-8")
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs)):        
        # Learning Rate Schedule setting
        # curr_lr = get_lr_schedule(epoch, start_epoch + num_epochs, scaled_lr, warm_up=True)
        curr_lr = get_lr_schedule(epoch, start_epoch + args.epochs, args.lr, args.lr_schedule_type, start_epoch=start_epoch)
        set_lr(optimizer, curr_lr)
        
        
        for i, (x_tgt, x_src, _, _, _) in enumerate(dl_train):
            x_tgt, x_src = x_tgt.cuda(), x_src.cuda()
            optimizer.zero_grad()
            
            loss = rf.forward(x_tgt, x_src) 
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if not args.dry_run:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": curr_lr,
                    "grad_norm": grad_norm.item(),
                    "epoch": epoch,
                })

        # Save checkpoint 
        if epoch % save_weight_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'max_train_accuracy' : curr_max_train_accuracy,
                'max_valid_accuracy' : curr_max_valid_accuracy,
            }
            
            # # Disabled due to memory issue
            # torch.save(checkpoint, os.path.join(weight_dir, f"checkpoint_epoch_{epoch}.pth"))
        
        # Evaluate performance
        if epoch % args.sample_freq != 0:
            continue

        rf.model.eval()
        with torch.no_grad():
            _, x_src_train, label_tgt_train, label_a_train, label_b_train = next(iter(dl_train))
            _, x_src_valid, label_tgt_valid, label_a_valid, label_b_valid = next(iter(dl_valid))
            batch_size_train = x_src_train.size(0)
            batch_size_valid = x_src_valid.size(0)
            x_tgt_train = torch.randn(batch_size_train, args.channel_size, args.image_size, args.image_size).cuda()
            x_tgt_valid = torch.randn(batch_size_valid, args.channel_size, args.image_size, args.image_size).cuda()
            x_src_train, label_tgt_train = x_src_train.cuda(), label_tgt_train.cuda()
            x_src_valid, label_tgt_valid = x_src_valid.cuda(), label_tgt_valid.cuda()
            
            samples_train = rf.sample(x_tgt_train, x_src_train)
            samples_valid = rf.sample(x_tgt_valid, x_src_valid)
            
            # Evaluate on Training data
            logits = classifier(samples_train[-1].detach())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            label_tgt = label_tgt_train.cpu().numpy()
            train_accuracy = (preds == label_tgt).mean()

            # Evaluate on Validation data
            logits = classifier(samples_valid[-1].detach())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
                            
            label_tgt = label_tgt_valid.cpu().numpy()
            valid_accuracy = (preds == label_tgt).mean()
        
            # Report on wandb
            if not args.dry_run:
                wandb.log({
                    "Train Accuracy": train_accuracy,
                    "Valid Accuracy": valid_accuracy,
                    "epoch": epoch,
                })
            
            # Save best performing checkpoint
            if valid_accuracy > 0.8 and valid_accuracy > curr_max_valid_accuracy:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'max_train_accuracy' : curr_max_train_accuracy,
                    'max_valid_accuracy' : curr_max_valid_accuracy,
                }
                f.write(f"epoch_{epoch}-valid_accy_{valid_accuracy:.2f}\n")
                f.flush()
                torch.save(checkpoint, os.path.join(weight_dir, f"model_best.pth"))        
                
            # Save image when [train, valid]-accuracy increase by 5%p or in save_weight_freq
            if (
                train_accuracy - curr_max_train_accuracy > 0.05 or \
                valid_accuracy - curr_max_valid_accuracy > 0.05 or \
                epoch % save_weight_freq == 0
            ):  
                save_sample_data(
                    args.dataset_name, args.p, x_src_train, samples_train[-1],
                    label_tgt_train, label_a_train, label_b_train,
                    result_dir, f"sample_{epoch}_train(trn_{train_accuracy:.2f}-val_{valid_accuracy:.2f}).png"
                )
                    
                curr_max_train_accuracy = max(curr_max_train_accuracy, train_accuracy)
                curr_max_valid_accuracy = max(curr_max_valid_accuracy, valid_accuracy)        
            
            # Early Stopping Exit!
            if valid_accuracy >= 0.99:
                tol += 1
                if tol > 25:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'max_train_accuracy' : curr_max_train_accuracy,
                        'max_valid_accuracy' : curr_max_valid_accuracy,
                    }
                    torch.save(checkpoint, os.path.join(weight_dir, f"model_final.pth"))
                    f.close()
                    
                    if not args.no_wandb_record:
                        wandb.finish()
                    exit()
            else:
                tol = 0
            
        rf.model.train()
        

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'max_train_accuracy' : curr_max_train_accuracy,
        'max_valid_accuracy' : curr_max_valid_accuracy,
    }
    torch.save(checkpoint, os.path.join(weight_dir, f"model_final.pth"))   
    f.close()
    
    if not args.dry_run:
        wandb.finish()
        