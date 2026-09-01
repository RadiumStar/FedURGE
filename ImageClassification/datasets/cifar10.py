"""
:file: cifar10.py
:date: 2025-07-28
:description: Loader for CIFAR-10 dataset
"""

from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10


CIFAR10_NUM_CHANNEL = 3
CIFAR10_NUM_CLASS = 10

CIFAR10_MEAN = (0.491400808095932, 0.48215898871421814, 0.44653093814849854)
CIFAR10_STD = (0.2023027539253235, 0.19941310584545135, 0.2009606957435608)

CIFAR10_ROOT = 'cifar10/'


def get_dataset(train=True, is_transform=True, root_folder='../../data/') -> CIFAR10:
    """Get CIFAR-10 dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :return: CIFAR-10 dataset
    """ 
    if is_transform: 
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
    else:
        transform = transforms.ToTensor()

    return CIFAR10(root=root_folder + CIFAR10_ROOT, train=train, download=True, transform=transform)