from classifier_models.resnet.manage_resnet_classifiers import get_resnet_emnist, get_resnet_kmnist


def get_classifier(dataset_name):
    # Data
    if dataset_name == 'emnist':
        return get_resnet_emnist()
    elif dataset_name == 'kmnist':
        return get_resnet_kmnist()
    elif dataset_name == 'two_dimensional':
        return None
        
    raise ValueError(f"Unexpected dataset_name : {dataset_name}")