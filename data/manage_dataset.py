import torch
import torch.nn.functional as F
from data.emnist.manage_emnist_dataset import get_confident_emnist_image_data_dict, EMNIST_PATH, save_sampled_mnist_images
from data.kmnist.manage_kmnist_dataset import get_confident_kmnist_image_data_dict, KMNIST_PATH
from data.two_dimensional.manage_two_dimensional_data import get_two_dimensional_image_data_dict, plot_2d_answer_and_pred, \
    squeeze_input_data, get_radii, RADIUS, NOISE_STD

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
        assert img_height == 1 and img_width == 1, f"Image size must be (1X1) but provided (img_height, img_width)=({img_height},{img_width})"
        return get_two_dimensional_image_data_dict(
            num_classes, num_images_per_class, data_radius=RADIUS, noise_std=NOISE_STD
        )
        
    raise ValueError(f"Unexpected dataset_name : {dataset_name}")


def save_sample_data(dataset_name, num_classes, x_src, sample, label_tgt, label_a, label_b, save_dir, file_name):
    if dataset_name in ('emnist', 'kmnist'):
        num_vis = 4
        save_sampled_mnist_images(x_src[:num_vis], sample[:num_vis], save_dir, file_name)
    elif dataset_name == 'two_dimensional':
        # sample = sample.cpu()
        # label_tgt = label_tgt.cpu()
        # radii = get_radii(squeeze_input_data(sample))
        plot_2d_answer_and_pred(sample, label_tgt, label_a, label_b, num_classes, RADIUS, save_dir=save_dir, file_name=file_name)
    else:
        raise NotImplementedError(f"Unexpected dataset_name : {dataset_name}")