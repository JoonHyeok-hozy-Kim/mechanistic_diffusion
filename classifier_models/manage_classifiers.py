from classifier_models.resnet.manage_resnet_classifiers import get_resnet_emnist


def get_classifier(dataset_name, num_classes):
    # Data
    if dataset_name == 'emnist':
        return get_resnet_emnist(num_classes, pretrained=True)
    elif dataset_name == 'kmnist':
        return None
    elif dataset_name == 'two_dimensional':
        return None
        
    raise ValueError(f"Unexpected dataset_name : {dataset_name}")