import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from tqdm import tqdm
import wandb

from data.emnist.manage_emnist_dataset import get_emnist_image_data_dict, MAX_EMNIST_NUM_CLASSES, EMNIST_PATH
from data.kmnist.manage_kmnist_dataset import get_kmnist_image_data_dict, MAX_KMNIST_NUM_CLASSES, KMNIST_PATH
from classifier_models.resnet.manage_resnet_classifiers import get_raw_resnet18, \
    EMNIST_CLASSIFIER_WEIGHT_PATH, KMNIST_CLASSIFIER_WEIGHT_PATH
import numpy as np


def train_resnet(data_source="emnist", train_fraction=0.8, dry_run=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 2000
    learning_rate = 0.001
    now_str = datetime.now().strftime("%y%m%d_%H%M")
    exp_name = f"train_resnet-data_soruce_{data_source}-train_fraction_{train_fraction}-lr_{learning_rate}"
    weight_dir = "weights"
    weight_dir = os.path.join(weight_dir, "classifier_weights_reserved")
    weight_dir = os.path.join(weight_dir, "resnet")
    os.makedirs(weight_dir, exist_ok=True)    
    

    if data_source == 'emnist':
        print("Loading EMNIST dataset...")
        num_classes = MAX_EMNIST_NUM_CLASSES
        full_dataset = get_emnist_image_data_dict(
            num_classes=num_classes, 
            data_root_dir=EMNIST_PATH,
            img_height=32, 
            img_width=32,
        )
        model = get_raw_resnet18(num_classes=num_classes)
        save_path = EMNIST_CLASSIFIER_WEIGHT_PATH
    elif data_source == 'kmnist':
        print("Loading kuzushiji emnist dataset...")
        num_classes = MAX_KMNIST_NUM_CLASSES
        full_dataset = get_kmnist_image_data_dict(
            num_classes=num_classes, 
            img_height=32, 
            img_width=32,
        )
        model = get_raw_resnet18(num_classes=num_classes)
        save_path = KMNIST_CLASSIFIER_WEIGHT_PATH
    else:
        raise f"Unknown data source : {data_source}"
    
    model = model.to(device=device)
    train_size = int(len(full_dataset) * train_fraction)
    val_size = len(full_dataset) - train_size
    
    # Shuffle labels and assign them to train/val sets
    ds_train, ds_val = random_split(full_dataset, [train_size, val_size])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    counter = 0
    patience = 10
    
    if not dry_run:
        wandb.init(project=f"train_resnet_mechansitic_diffusion", name=exp_name) 

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dl_train, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if not dry_run:
                wandb.log({"train_loss": loss.item()})
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=train_loss/len(dl_train), acc=100.*correct/total)
            
        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in dl_val:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        if not dry_run:
            wandb.log({"Valid Accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    if not dry_run:
        wandb.finish()
    

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_source", type=str, choices=['emnist', 'kmnist'], default='emnist')
        parser.add_argument("--train_fraction", type=float, default=0.8)
        parser.add_argument("--dry_run", action="store_true", help="Choose to run in local.")
        return parser.parse_args()
    
    args = parse_args()
    
    train_resnet(
        data_source=args.data_source,
        train_fraction=args.train_fraction,
        dry_run=args.dry_run,
    )