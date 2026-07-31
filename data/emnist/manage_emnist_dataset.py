import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

MAX_EMNIST_NUM_CLASSES = 35     # Max num classes excluding 'O' (cf. Duplicacy with '0')

def get_confident_emnist_image_data_dict(num_classes, num_images_per_class, classifier, confidence_threshold, data_root_dir, img_height, img_width):
    ds_emnist_transformed_all = get_emnist_image_data_dict(num_classes, data_root_dir, img_height, img_width)
    
    raise NotImplementedError("Needs further implementation")


def get_emnist_image_data_dict(num_classes, data_root_dir, img_height, img_width):
    ds_emnist_raw_balanced = torchvision.datasets.EMNIST(   # (B, H, W) where H=W=28
        root=data_root_dir, 
        split='balanced', 
        train=True, 
        download=True, 
    )    
    label_index_dict = get_emnist_dataset_index(ds_emnist_raw_balanced, data_root_dir)
    
    all_images = []
    all_labels = []
    
    # Use alphabets only.
    if num_classes <= 26:
        num_arithmetics = 10
        target_labels = list(range(num_arithmetics, num_arithmetics + num_classes)) # Exclude arithmetic characters : 0~9
        label_map = {old_label: new_label for new_label, old_label in enumerate(target_labels)}

        for old_label in range(num_arithmetics, num_arithmetics + num_classes):
            if old_label in label_index_dict:
                target_indices_by_label = label_index_dict[old_label]
                
                imgs = ds_emnist_raw_balanced.data[target_indices_by_label].cpu()                   # (B, H, W)    where H=W=28
                imgs_transformed = apply_batch_transform(imgs, img_height, img_width, swap_hw=True) # (B, C, W, H) where C=1, H=W=32
                
                new_label = label_map[old_label]
                labels = torch.full((len(imgs_transformed),), new_label, dtype=torch.long)  # (B, )
                
                all_images.append(imgs_transformed)
                all_labels.append(labels)
    # Use numeric digit as well. Exclude 'O' considering similarity with '0'
    elif num_classes <= MAX_EMNIST_NUM_CLASSES:
        capital_O_idx = 10 + int(ord('O') - ord('A'))
        for label_idx in range(num_classes):
            adj_label_idx = label_idx if label_idx < capital_O_idx else label_idx + 1
            target_indices_by_label = label_index_dict[adj_label_idx]
            
            imgs = ds_emnist_raw_balanced.data[target_indices_by_label].cpu()                   # (B, H, W)    where H=W=28
            imgs_transformed = apply_batch_transform(imgs, img_height, img_width, swap_hw=True) # (B, C, W, H) where C=1, H=W=32
            
            labels = torch.full((len(imgs_transformed),), label_idx, dtype=torch.long)  # (B, )
            
            all_images.append(imgs_transformed)
            all_labels.append(labels)
    else:
        raise ValueError(f"num_classes greater than {MAX_EMNIST_NUM_CLASSES}. (input num_clases={num_classes})")

    
    final_images = torch.cat(all_images, dim=0) # (N, C, H, W) where N = B * num_classes
    final_labels = torch.cat(all_labels, dim=0) # (N, )
    full_dataset = TensorDataset(final_images, final_labels)
    
    return full_dataset



def get_emnist_dataset_index(dataset, emnist_path, split='balanced'):
    index_path = os.path.join(emnist_path, 'index')
    index_file_name = os.path.join(index_path, f'emnist_index_{split}.txt')
    
    if os.path.exists(index_file_name):
        with open(index_file_name, "r") as f:
            raw_index = json.load(f)
            print(f"Loaded existing EMNIST index from {index_file_name}")
            return {int(k): v for k, v in raw_index.items()}
                
    print(f"Existing EMNIST index not found. Generating new index at {index_file_name}...", end=" ")
    os.makedirs(index_path, exist_ok=True)
    index_result = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        index_result[label].append(idx)
        
    with open(index_file_name, "w") as f:
        json.dump(index_result, f)
    
    print("fin.")
    return {int(k): v for k, v in index_result.items()}



def apply_batch_transform(raw_tensor, img_height, img_width, swap_hw=True):
    
    if raw_tensor.ndim == 3: # Single image case : (C, H, W) -> (1, C, H, W)
        x = raw_tensor.unsqueeze(1)  # (B, C, H, W)
    else:
        x = raw_tensor  # (B, C, H, W)
    
    x = x.float()
    x = F.interpolate(x, size=(img_height, img_width), mode='bilinear', align_corners=False)
    if swap_hw:
        x = x.permute(0, 1, 3, 2)   # (B, C, H, W) -> (B, C, W, H)
    x = (x / 127.5) - 1.0  # Normalize to [-1, 1]
    return x

