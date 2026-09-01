"""
File: __init__.py
Date: 2025-07-10
Description: learning models module
"""

from typing import Optional


# from .logistic_regression import LogisticRegression_BINARY_MNIST, LogisticRegression_MNIST
from .mlp import MLP_MNIST
from .cnn import CNN_MNIST, CNN_FASHION_MNIST, LeNet5_MNIST
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet34_CIFAR100, ResNet34_TINY_IMAGENET 


def get_model_class(model_name: str, dataset_name: Optional[str] = None):
    """Get model by name

    :param model_name: name of model
    :param dataset_name: name of dataset, defaults to None
    :return: model class or function
    :raises ValueError: if model not found
    """
    
    if dataset_name is None:
        ret_model_func = model_name
    else:
        ret_model_func = model_name + f"_{dataset_name.upper()}"

    try: 
        # return eval(f"{ret_model_func}({input_channel}, {input_dim}, {num_class})")
        return eval(f"{ret_model_func}")
    except NameError:
        raise ValueError(f"Model {model_name} for dataset {dataset_name} not found. Please check the model name and dataset name. Available models: ")
