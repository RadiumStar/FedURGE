"""
:file: tiny_imagenet.py
:date: 2025-07-30
:description: Loader for Tiny ImageNet dataset
"""

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TINY_IMAGENET_NUM_CHANNEL = 3
TINY_IMAGENET_NUM_CLASS = 200

TINY_IMAGENET_MEAN = (0.480, 0.448, 0.397)
TINY_IMAGENET_STD = (0.276, 0.269, 0.282)

TINY_IMAGENET_ROOT = 'tiny-imagenet-200/'


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        self._prepare_data()

    def _prepare_data(self):
        self.data = []
        self.targets = []
        self.img_paths = []

        wnids_path = os.path.join(self.root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]

        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        if self.train:
            for wnid in self.wnids:
                img_dir = os.path.join(self.root, 'train', wnid, 'images')
                for fname in os.listdir(img_dir):
                    if fname.endswith('.JPEG'):
                        path = os.path.join(img_dir, fname)
                        self.img_paths.append(path)
                        self.targets.append(self.wnid_to_idx[wnid])
                        img = Image.open(path).convert('RGB')
                        self.data.append(np.array(img))
        else:
            val_img_path = os.path.join(self.root, 'val', 'images')
            val_annotations_path = os.path.join(self.root, 'val', 'val_annotations.txt')

            img_to_wnid = {}
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split('\t')
                    img_to_wnid[tokens[0]] = tokens[1]

            for fname in os.listdir(val_img_path):
                if fname.endswith('.JPEG'):
                    wnid = img_to_wnid[fname]
                    if wnid not in self.wnid_to_idx:
                        continue
                    path = os.path.join(val_img_path, fname)
                    self.img_paths.append(path)
                    self.targets.append(self.wnid_to_idx[wnid])
                    img = Image.open(path).convert('RGB')
                    self.data.append(np.array(img))

        self.data = np.stack(self.data, axis=0)  # (N, 64, 64, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])  # back to PIL for transform
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target


def get_dataset(train=True, is_transform=True, root_folder='../../data/'):
    """Get Tiny ImageNet dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :param root_folder: Root folder for the dataset, defaults to '../../data/'
    :return: Tiny ImageNet dataset
    """
    split_dir = 'train' if train else 'val'

    if is_transform:
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
                # Optional: Cutout / RandomErasing
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
            ])
    else:
        transform = transforms.ToTensor()

    return TinyImageNet(root=root_folder + TINY_IMAGENET_ROOT, train=train, transform=transform)



if __name__ == "__main__": 
    dataset = get_dataset(train=True, is_transform=True)
    print(f"Loaded Tiny ImageNet dataset with {len(dataset)} samples.")
    print(f"Number of classes: {TINY_IMAGENET_NUM_CLASS}")
    print(f"Image shape: {dataset.data[0].shape}")
    print(f"First target: {dataset.targets[0]}")
    # print the wnid, idx and the corresponding class name for the first 10 samples
    for i in range(10):
        wnid = dataset.wnids[i]
        idx = dataset.wnid_to_idx[wnid]
        print(f"Sample {i}: wnid={wnid}, idx={idx}")
