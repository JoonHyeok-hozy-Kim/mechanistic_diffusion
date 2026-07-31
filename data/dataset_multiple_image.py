from collections import defaultdict
from random import shuffle, randint
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import numpy as np


class ResNet_SingleToken_MultipleImages(Dataset):
    def __init__(self, p, num_images, letter_images):
        self.p = p
        self.num_images = num_images
        self.letter_images = letter_images  # Receive Resnet Images directly
        
        self.data = [
            (a, ndx, -1) for a in range(p) for ndx in range(num_images)
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        a, ndx, _ = self.data[idx]
        assert ndx < len(self.letter_images[a]), \
            f"Less available EMNIST images for label {a} ({len(self.letter_images[a])}) \
                than the num_images ({self.num_images})"
        
        img_a = self.letter_images[a][ndx]
        img_b = torch.rand_like(img_a)          # Noise on b token
        img_result = torch.rand_like(img_a)     # Noise on c token
        assert img_a.shape == img_b.shape == img_result.shape == (1, 32, 32)
        
        img_tgt = img_result  # Shape: (1, 32, 32)
        # Concatenate along width dimension (dim=2) to get shape (1, 32, 64)
        img_src = torch.cat([img_a, img_b], dim=2)
        assert img_src.shape == (1, 32, 64), f"Expected img_src shape (1, 32, 64), got {img_src.shape}"
        label_tgt = None
        label_a = torch.tensor(a, dtype=torch.long)
        label_b = None
        return img_tgt, img_src, label_a
    
    def get_num_data(self, idx):
        return self.data[idx]
    


class AllImage_ResNet_ModularDataset(Dataset):
    def __init__(self, p, num_images, letter_images):
        self.p = p
        self.num_images = num_images
        self.letter_images = letter_images  # Receive Resnet Images directly        
        self.batch_repeat = num_images // self.p
        
        self.data = []
        
        image_pool_a = defaultdict(list)
        image_pool_b = defaultdict(list)
        for p_i in range(p):
            for n_i in range(num_images):
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
        c = (a+b) % self.p
        
        img_a = self.letter_images[p_a_key[0]][p_a_key[1]]
        img_b = self.letter_images[p_b_key[0]][p_b_key[1]]
        
        assert img_a.shape == img_b.shape == (1, 32, 32)
        
        img_tgt = torch.randn_like(img_a)  # Shape: (1, 32, 32)
        # Concatenate along width dimension (dim=2) to get shape (1, 32, 64)
        img_src = torch.cat([img_a, img_b], dim=2)
        assert img_src.shape == (1, 32, 64), f"Expected img_src shape (1, 32, 64), got {img_src.shape}"
        label_tgt = torch.tensor(c, dtype=torch.long)
        label_a = torch.tensor(a, dtype=torch.long)
        label_b = torch.tensor(b, dtype=torch.long)
        return img_tgt, img_src, label_tgt, label_a, label_b
    
    def get_num_data(self, idx):
        a, b, p_a_key, p_b_key = self.data[idx]
        c = (a+b) % self.p
        return [a, b, c]


class AllImage_ResNet_ResultImage_ModularDataset(AllImage_ResNet_ModularDataset):
    def __init__(self, p, num_images, letter_images, correct_result_img=True):
        super().__init__(p, num_images, letter_images)
        self.correct_result_img = correct_result_img        
        
    def __getitem__(self, idx):
        a, b, p_a_key, p_b_key = self.data[idx]
        truc_c = (a+b) % self.p
        if self.correct_result_img:
            ret_c = truc_c
        else:
            offset = randint(1, self.p-1)
            ret_c = (truc_c + offset) % self.p  
        
        img_a = self.letter_images[p_a_key[0]][p_a_key[1]]
        img_b = self.letter_images[p_b_key[0]][p_b_key[1]]
        
        assert img_a.shape == img_b.shape == (1, 32, 32)
        
        avaliable_c_images = len(self.letter_images[ret_c])
        c_idx = randint(0, avaliable_c_images-1)
        img_tgt = self.letter_images[ret_c][c_idx]  # Shape: (1, 32, 32)
        
        # Concatenate along width dimension (dim=2) to get shape (1, 32, 64)
        img_src = torch.cat([img_a, img_b], dim=2)
        assert img_src.shape == (1, 32, 64), f"Expected img_src shape (1, 32, 64), got {img_src.shape}"
        label_tgt = torch.tensor(ret_c, dtype=torch.long)
        label_a = torch.tensor(a, dtype=torch.long)
        label_b = torch.tensor(b, dtype=torch.long)
        return img_tgt, img_src, label_tgt, label_a, label_b

        
    