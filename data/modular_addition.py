import torch
from torch.utils.data import Dataset
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from random import shuffle, randint


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
            np.random.seed(42)
            group_keys = list(pair_groups.keys())
            np.random.shuffle(group_keys)
            train_size = int(train_fraction * len(group_keys))
            
            # Collect pairs based on group assignment
            if split == 'train':
                train_groups = set(group_keys[:train_size])
                self.data = [pair for key, group in pair_groups.items() if key in train_groups for pair in group]
            else:
                val_groups = set(group_keys[train_size:])
                self.data = [pair for key, group in pair_groups.items() if key in val_groups for pair in group]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        actual_index = self._get_actual_index(idx)
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
    
    def _get_actual_index(self, idx):
        return idx % len(self.data)
    
    def get_num_data(self, idx):
        actual_idx = self._get_actual_index(idx)
        return self.data[actual_idx]


class Sorted_Modular_Arithmetic_Dataset(Modular_Arithmetic_Dataset, ABC):
    def __init__(self, p=23, split='train', train_fraction=0.3, num_images_per_class=1, img_data_dict=None):
        super().__init__(p, split, train_fraction, num_images_per_class, img_data_dict)
        self.grids = {'super': {}, 'self': {}}
        self._generate_grid()
    
    
    def _generate_grid(self):
        pass
    
    
    def _get_super_idx(self, idx):
        idx = self._get_actual_index(idx)
        pair = self.grids['self'][idx]
        return self.grids['super'][pair]        
    
    
    def __getitem__(self, idx):
        return super().__getitem__(self._get_super_idx(idx))
    
    
    def get_num_data(self, idx):
        return super().get_num_data(self._get_super_idx(idx))


class Operand_Sorted_Modular_Arithmetic_Dataset(Sorted_Modular_Arithmetic_Dataset):
    def _generate_grid(self):
        for i_super, (a, b, res) in enumerate(self.data):
            self.grids['super'][(a, b)] = i_super
        
        i_self = 0
        for a in range(self.p):
            for b in range(self.p):
                self.grids['self'][i_self] = (a, b)
                i_self += 1    


class Result_Sorted_Modular_Arithmetic_Dataset(Sorted_Modular_Arithmetic_Dataset):
    def _generate_grid(self):
        for i_super, (a, b, res) in enumerate(self.data):
            self.grids['super'][(a, b)] = i_super
        
        i_self = 0
        for c in range(self.p):
            for a in range(self.p):
                b = c - a
                if b < 0:
                    b += self.p
                self.grids['self'][i_self] = (a, b)
                i_self += 1


class Inference_All_Image_Modular_Arithmetic_Dataset(Dataset):
    '''
    Generate all possible N^2 pairs of operand images a and b for inference only.
    When `correct_result_img` is true, it switches the result image to wrong one for the rectification
    '''
    def __init__(self, p, num_images_per_class, img_data_dict, correct_result_img=True):
        self.p = p
        self.num_images_per_class = num_images_per_class
        self.img_data_dict = img_data_dict     
        self.correct_result_img = correct_result_img
        self.batch_repeat = num_images_per_class // self.p        
        self.data = []
        
        image_pool_a = defaultdict(list)
        image_pool_b = defaultdict(list)
        for p_i in range(p):
            for n_i in range(num_images_per_class):
                image_pool_a[p_i].append((p_i, n_i))
                image_pool_b[p_i].append((p_i, n_i))
        for p_i, L in image_pool_a.items():
            shuffle(L)
            shuffle(image_pool_b[p_i])
        
        has_data = True
        while has_data:
            pairs = []
            for p_a in range(p):
                if len(image_pool_a[p_a]) == 0 or not has_data:
                    has_data = False
                    break
                p_a_key = image_pool_a[p_a].pop(0)
                for p_b in range(p):
                    if len(image_pool_b[p_b]) == 0:
                        has_data = False
                        break                
                    p_b_key = image_pool_b[p_b].pop(0)
                    pairs.append([p_a, p_b, p_a_key, p_b_key])
            if len(pairs) == p**2:
                self.data.extend(pairs)
        
        assert len(self.data) // (p**2) == self.batch_repeat, f"Batch repeat ({self.batch_repeat}) does not comply with the actual data size ({len(self.data) // (p**2)}*{(p**2)})."
    
    def __len__(self):
        return len(self.data)  
        
    def __getitem__(self, idx):
        a, b, p_a_key, p_b_key = self.data[idx]
        truc_c = (a+b) % self.p
        if self.correct_result_img:
            ret_c = truc_c
        else:
            # Randomly chooose the wrong result label
            offset = randint(1, self.p-1)
            ret_c = (truc_c + offset) % self.p  
        
        img_a = self.img_data_dict[p_a_key[0]][p_a_key[1]]
        img_b = self.img_data_dict[p_b_key[0]][p_b_key[1]]
        
        assert img_a.shape == img_b.shape == (1, 32, 32)
        
        avaliable_c_images = len(self.img_data_dict[ret_c])
        c_idx = randint(0, avaliable_c_images-1)
        img_tgt = self.img_data_dict[ret_c][c_idx]  # Shape: (1, 32, 32)
        
        # Concatenate along width dimension (dim=2) to get shape (1, 32, 64)
        img_src = torch.cat([img_a, img_b], dim=2)
        # assert img_src.shape == (1, 32, 64), f"Expected img_src shape (1, 32, 64), got {img_src.shape}"
        label_tgt = torch.tensor(ret_c, dtype=torch.long)
        label_a = torch.tensor(a, dtype=torch.long)
        label_b = torch.tensor(b, dtype=torch.long)
        return img_tgt, img_src, label_tgt, label_a, label_b
    
    def get_num_data(self, idx):
        a, b, p_a_key, p_b_key = self.data[idx]
        c = (a+b) % self.p
        return [a, b, c]