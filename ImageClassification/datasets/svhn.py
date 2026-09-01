"""
:file: svhn.py
:date: 2025-07-28
:description: Loader for SVHN dataset
"""


from torchvision import datasets, transforms
from torchvision.datasets import SVHN
import numpy as np
from PIL import Image


SVHN_NUM_CHANNEL = 3
SVHN_NUM_CLASS = 10

SVHN_MEAN = (0.4376821, 0.4437697, 0.47280442) 
SVHN_STD = (0.19803012, 0.20101562, 0.19703614)

SVHN_ROOT = 'svhn/'


class Aligned_SVHN(SVHN):
    def __getitem__(self, i):
        img, target = self.data[i], int(self.labels[i])

        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_dataset(train=True, is_transform=True, root_folder='../../data/') -> Aligned_SVHN:
    """Get SVHN dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :return: SVHN dataset
    """
    if is_transform:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ])
    else:
        transform = transforms.ToTensor()

    data = Aligned_SVHN(root=root_folder + SVHN_ROOT, split='train' if train else 'test', download=True, transform=transform)
    data.data = np.transpose(data.data, (0, 2, 3, 1))  # CHW -> HWC
    data.targets = data.labels  # For SVHN, labels are stored in 'labels' attribute
    return data