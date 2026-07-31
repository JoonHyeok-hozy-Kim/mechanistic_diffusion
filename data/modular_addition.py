import torch
from torch.utils.data import Dataset
import numpy as np


class Modular_Arithmetic_Dataset(Dataset):
    def __init__(self, p=23, split='train', train_fraction=0.3, num_images_per_class=1, img_data_dict=None):
        """
        Font-based modular arithmetic dataset with multiple font variations.
        
        Args:
            p: Modulus (should be <= 23 for all characters: A-Z, a-z, 0-9, special chars)
            split: 'train' or 'val'
            train_fraction: Fraction of data to use for training
            num_images: Number of different font variations per character
            img_dir: Directory containing generated font images
            fonts_dir: Directory containing TTF font files (required if num_images > 1)
        """
        assert img_data_dict is not None, f"img_data_dict is not provided."
        self.p = p
        self.num_images_per_class = num_images_per_class
        self.img_data_dict = img_data_dict
            
        # Generate all possible pairs grouped by symmetric pairs
        # Group (a, b) and (b, a) together to avoid permutation-only learning
        pair_groups = {}
        for a in range(p):
            for b in range(p):
                result = (a + b) % p
                # Use canonical form (min(a,b), max(a,b)) as key for grouping
                key = (min(a, b), max(a, b))
                if key not in pair_groups:
                    pair_groups[key] = []
                pair_groups[key].append((a, b, result))
        
        # Convert groups to list of pairs, maintaining symmetry
        all_pairs = []
        for group in pair_groups.values():
            all_pairs.extend(group)
                    
        if split == 'all':
            self.data = all_pairs            
        else:        
            # Split into train/val at the group level to maintain symmetry
            group_keys = list(pair_groups.keys())
            np.random.shuffle(group_keys)
            train_size = int(train_fraction * len(group_keys))
            
            # Collect pairs based on group assignment
            if split == 'train':
                train_groups = set(group_keys[:train_size])
                self.data = [pair for key, group in pair_groups.items() 
                            if key in train_groups for pair in group]
            else:
                val_groups = set(group_keys[train_size:])
                self.data = [pair for key, group in pair_groups.items() 
                            if key in val_groups for pair in group]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        actual_index = self.get_actual_index(idx)
        a, b, result = self.data[actual_index]
        
        # Randomly sample font variation for each character independently
        font_idx_a = np.random.randint(0, self.num_images_per_class)
        font_idx_b = np.random.randint(0, self.num_images_per_class)
        font_idx_result = np.random.randint(0, self.num_images_per_class)
        
        img_a = self.img_data_dict[a][font_idx_a]
        img_b = self.img_data_dict[b][font_idx_b]
        img_result = self.img_data_dict[result][font_idx_result]
        # assert img_a.shape == img_b.shape == img_result.shape == (1, 32, 32)
        assert img_a.shape == img_b.shape == img_result.shape 
        
        img_tgt = img_result  # Shape: (1, 32, 32)
        # Concatenate along width dimension (dim=2) to get shape (1, 32, 64)
        img_src = torch.cat([img_a, img_b], dim=2)
        # assert img_src.shape == (1, 32, 64), f"Expected img_src shape (1, 32, 64), got {img_src.shape}"
        
        label_tgt = torch.tensor(result, dtype=torch.long)
        label_a = torch.tensor(a, dtype=torch.long)
        label_b = torch.tensor(b, dtype=torch.long)
        return img_tgt, img_src, label_tgt, label_a, label_b
    
    def get_actual_index(self, idx):
        return idx % len(self.data)
    
    def get_num_data(self, idx):
        actual_idx = self.get_actual_index(idx)
        return self.data[actual_idx]
    