from classifier_models.resnet.manage_resnet_classifiers import get_resnet_emnist, get_resnet_kmnist
from classifier_models.non_training.two_dimensional_classifier import Two_Dimensional_Data_Classifier

def get_classifier(dataset_name, num_classes=None):
    # Data
    if dataset_name == 'emnist':
        return get_resnet_emnist()
    elif dataset_name == 'kmnist':
        return get_resnet_kmnist()
    elif dataset_name == 'two_dimensional':
        assert num_classes is not None, f"Number of classes(p) should be provided for the two dimensional classifier."
        return Two_Dimensional_Data_Classifier(num_classes)
        
    raise ValueError(f"Unexpected dataset_name : {dataset_name}")