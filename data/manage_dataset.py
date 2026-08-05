import torch.nn.functional as F
from data.emnist.manage_emnist_dataset import get_confident_emnist_image_data_dict, EMNIST_PATH
from data.kmnist.manage_kmnist_dataset import get_confident_kmnist_image_data_dict, KMNIST_PATH

def get_confident_image_data_dict(dataset_name, num_classes, num_images_per_class, classifier, confidence_threshold, img_height=32, img_width=32):
    # Data
    if dataset_name == 'emnist':
        return get_confident_emnist_image_data_dict(
            num_classes, num_images_per_class, 
            classifier, confidence_threshold, 
            data_root_dir=EMNIST_PATH, img_height=img_height, img_width=img_width,
        )
    elif dataset_name == 'kmnist':
        return get_confident_kmnist_image_data_dict(
            num_classes, num_images_per_class, 
            classifier, confidence_threshold, 
            img_height=img_height, img_width=img_width,
        )
    elif dataset_name == 'two_dimensional':
        return None
        
    raise ValueError(f"Unexpected dataset_name : {dataset_name}")