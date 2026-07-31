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

from data.emnist.manage_emnist_dataset import get_emnist_image_data_dict
# from data.kuzushiji.handle_kujushiji import get_kuzushiji_dataset_by_num_classes
from classifier_models.resnet.manage_resnet_classifiers import get_resnet18_emnist
import numpy as np


def train_resnet(data_source="emnist", train_fraction=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 2000
    learning_rate = 0.001
    now_str = datetime.now().strftime("%y%m%d_%H%M")
    exp_name = f"train_resnet-data_soruce_{data_source}-train_fraction_{train_fraction}-lr_{learning_rate}"
    weight_dir = "weights"
    weight_dir = os.path.join(weight_dir, "classifier_resnet_reserved")
    weight_dir = os.path.join(weight_dir, "classifier_resnet")
    os.makedirs(weight_dir, exist_ok=True)    

    if data_source == 'emnist':
        print("Loading EMNIST dataset...")
        # full_emnist_dataset = get_emnist_balanced_transformed(emnist_path="data/emnist")
        full_dataset = get_emnist_image_data_dict(
            num_classes=35, 
            data_root_dir="data/emnist", 
            img_height=32, 
            img_width=32,
        )
    elif data_source == 'kuzushiji':
        print("Loading kuzushiji emnist dataset...")
        # full_dataset = get_kuzushiji_dataset_by_num_classes(num_classes)
        raise NotImplementedError("Needs further implementation")
    else:
        raise f"Unknown data source : {data_source}"
    
    train_size = int(len(full_dataset) * train_fraction)
    val_size = len(full_dataset) - train_size
    
    # Shuffle labels and assign them to train/val sets
    ds_train, ds_val = random_split(full_dataset, [train_size, val_size])

    # 4. DataLoader에 바로 전달
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_resnet18_emnist(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    counter = 0
    patience = 10
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
        wandb.log({"Valid Accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(weight_dir, f"resnet_{data_source}_best.pt")
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print("Early stopping triggered.")
            break
        
    wandb.finish()
    

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_source", type=str, default='kuzushiji')
        parser.add_argument("--train_fraction", type=float, default=0.8)
        return parser.parse_args()
    
    args = parse_args()
    
    train_resnet(
        data_source=args.data_source,
        train_fraction=args.train_fraction,
    )