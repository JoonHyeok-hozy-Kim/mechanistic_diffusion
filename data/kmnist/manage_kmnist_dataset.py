import os
import kagglehub
import json
from collections import defaultdict
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader

KMNIST_PATH = 'data/kmnist'
KMNIST_RAW_DATA_PATH = os.path.join(KMNIST_PATH, 'KMNIST')
MAX_KMNIST_NUM_CLASSES = 47


class RelabeledSubset(torch.utils.data.Dataset): 
    def __init__(self, dataset, indices, class_to_new_idx):
        self.dataset = dataset
        self.indices = indices
        self.class_to_new_idx = class_to_new_idx
        
        print(f"[DEBUG] New labels count: {len(self.class_to_new_idx)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, old_label = self.dataset[actual_idx]
        
        class_name = self.dataset.classes[old_label]
        new_label = self.class_to_new_idx[class_name]
        
        return img, new_label


def get_confident_kmnist_image_data_dict(num_classes, num_images_per_class, classifier, confidence_threshold, img_height, img_width):
    device = next(classifier.parameters()).device
    
    ds_kmnist_transformed_all = get_kmnist_image_data_dict(num_classes, img_height, img_width)
    dl_kmnist_transformed_all = DataLoader(ds_kmnist_transformed_all, batch_size=128, shuffle=False)

    all_confidences = []
    all_is_correct = []
    all_labels = []
    
    # Get confidences on the true label and correct prediction masks
    with torch.no_grad():
        for images, labels in tqdm(dl_kmnist_transformed_all, desc="Calculating Confidence"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier(images)
            probs = torch.softmax(outputs, dim=1)
            
            confidences = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            predictions = torch.argmax(probs, dim=1)
            is_correct = (predictions == labels)

            all_confidences.append(confidences.cpu())
            all_is_correct.append(is_correct.cpu())
            all_labels.append(labels.cpu())
    
    final_confidences = torch.cat(all_confidences)
    final_is_correct = torch.cat(all_is_correct)
    true_labels = torch.cat(all_labels)
    
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
        for idx in indices:
            img, _ = ds_kmnist_transformed_all[idx.item()]
            curr_list.append(img)
        letter_images[label] = curr_list
    
    return letter_images


def get_kmnist_image_data_dict(num_classes, img_height, img_width):
    raw_dataset, raw_class_cnt = get_raw_kmnist_image_data(img_height, img_width)
    
    sorted_classes = sorted(raw_class_cnt, key=lambda x: x[1], reverse=True)
    
    sorted_names = [name for name, num in sorted_classes[:num_classes]]    
    assert len(sorted_names) == num_classes, f"Less classes available {len(sorted_names)} (Requested {num_classes})!"    
    
    name_to_new_idx = {name: i for i, name in enumerate(sorted_names)}
    set_sorted_names = set(sorted_names)
    target_indices = [
        i for i, label_idx in enumerate(raw_dataset.targets)
        if raw_dataset.classes[label_idx] in set_sorted_names
    ]
    
    filtered_dataset = RelabeledSubset(raw_dataset, target_indices, name_to_new_idx)
    
    return filtered_dataset
        

def get_raw_kmnist_image_data(img_height, img_width):
    kanji_path = os.path.join(KMNIST_RAW_DATA_PATH, "kkanji")
    kanji_path = os.path.join(kanji_path, "kkanji2")
    if not os.path.exists(kanji_path):
        down_load_kmnist(KMNIST_RAW_DATA_PATH)

    class_counts = []
    for class_name in os.listdir(kanji_path):
        class_path = os.path.join(kanji_path, class_name)
        if os.path.isdir(class_path):
            num_files = len(os.listdir(class_path))
            class_counts.append((class_name, num_files))

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=kanji_path, transform=transform)

    return full_dataset, class_counts


def down_load_kmnist(download_path):
    downloaded_path = kagglehub.dataset_download("anokas/kuzushiji", force_download=True, output_dir=download_path)
    print(f'KMNIST data successfully downloaded at : {downloaded_path}')