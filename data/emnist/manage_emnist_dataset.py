import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as tvu

from tqdm import tqdm
import matplotlib.pyplot as plt

EMNIST_PATH = 'data/emnist'
MAX_EMNIST_NUM_CLASSES = 35     # Max num classes excluding 'O' (cf. Duplicacy with '0')


def get_confident_emnist_image_data_dict(num_classes, num_images_per_class, classifier, confidence_threshold, data_root_dir, img_height, img_width):
    device = next(classifier.parameters()).device
    
    ds_emnist_transformed_all = get_emnist_image_data_dict(num_classes, data_root_dir, img_height, img_width)
    dl_emnist_transformed_all = DataLoader(ds_emnist_transformed_all, batch_size=128, shuffle=False)

    all_confidences = []
    all_is_correct = []
    
    # Get confidences on the true label and correct prediction masks
    with torch.no_grad():
        for images, labels in tqdm(dl_emnist_transformed_all, desc="Calculating Confidence"):
            images = images.to(device)
            outputs = classifier(images)     # (B, num_classes)
            probs = torch.softmax(outputs, dim=1)   # (B, num_classes)
            
            confidences = probs.gather(1, labels.to(device).unsqueeze(1)).squeeze(1)  # (B, ) : confidence for the true class
            predictions = torch.argmax(probs, dim=1)    # (B, )           
                
            is_correct = (predictions == labels.to(device))
            all_confidences.append(confidences.cpu())
            all_is_correct.append(is_correct.cpu())
    
    final_confidences = torch.cat(all_confidences)
    final_is_correct = torch.cat(all_is_correct)
    true_labels = ds_emnist_transformed_all.tensors[1]
    
    top_k_indices_by_label = {}
    for class_idx in range(num_classes):
        class_mask = (true_labels == class_idx) & final_is_correct & (final_confidences >= confidence_threshold)
        class_indices = torch.where(class_mask)[0]
        class_confidences = final_confidences[class_indices]
        
        # Top-k among class indices!
        assert num_images_per_class <= len(class_confidences), f"Not enough confident images for class {class_idx}: required {num_images_per_class}, found {len(class_confidences)}"
        _, topk_indices = torch.topk(class_confidences, k=num_images_per_class)        
        top_k_indices_by_label[class_idx] = class_indices[topk_indices]
    
    # List of single image tensors
    letter_images = {}
    for label, indices in top_k_indices_by_label.items():
        curr_list = []
        for img in ds_emnist_transformed_all.tensors[0][indices]:
            curr_list.append(img)
        letter_images[label] = curr_list
    
    return letter_images


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
    idx_O = 10 + int(ord('O') - ord('A'))
    
    # Use alphabets only.
    if num_classes > 35:
        raise ValueError(f"num_classes greater than {MAX_EMNIST_NUM_CLASSES}. (input num_clases={num_classes})")
    
    for label_idx in range(num_classes):
        # adj_label_idx = label_idx + 1       # Skip '0' for confusion with 'O'     
           
        # Skip 'O' for confusion with '0'
        if label_idx >= idx_O:
            adj_label_idx = label_idx + 1
        else:
            adj_label_idx = label_idx
        
        target_indices_by_label = label_index_dict[adj_label_idx]
        imgs = ds_emnist_raw_balanced.data[target_indices_by_label].cpu()                    # (B, W, H)    where H=W=28
        imgs_transformed = apply_batch_transform(imgs, img_height, img_width, swap_hw=True)  # (B, C, H, W) where C=1, H=W=32
        labels = torch.full((len(imgs_transformed),), label_idx, dtype=torch.long)  # (B, )
        all_images.append(imgs_transformed)
        all_labels.append(labels)
    
    final_images = torch.cat(all_images, dim=0) # (N, C, H, W) where N = B * num_classes
    final_labels = torch.cat(all_labels, dim=0) # (N, )
    full_dataset = TensorDataset(final_images, final_labels)
    
    return full_dataset



def get_emnist_dataset_index(dataset, data_root_dir, split='balanced'):
    index_path = os.path.join(data_root_dir, 'index')
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


def save_sampled_mnist_images(x_src, sample, save_dir, file_name):
    concat_img = torch.cat([x_src, sample], dim=3)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    tvu.save_image(concat_img, file_path, nrow=1)