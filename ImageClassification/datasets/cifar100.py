"""
:file: cifar100.py
:date: 2025-07-28
:description: Loader for CIFAR-100 dataset
"""


from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100


CIFAR100_NUM_CHANNEL = 3
CIFAR100_NUM_CLASS = 100

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR100_ROOT = 'cifar100/'


def get_dataset(train=True, is_transform=True, root_folder='../../data/') -> CIFAR100:
    """Get CIFAR-100 dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :return: CIFAR-100 dataset
    """ 
    if is_transform:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])
    else:
        transform = transforms.ToTensor()

    return CIFAR100(root=root_folder + CIFAR100_ROOT, train=train, download=True, transform=transform)