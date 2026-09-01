""" 
:file: loader.py
:date: 2026-08-04 (create date) / 2026-08-04 (last modified date)
:description: Load Binary MNIST dataset for distributed logistic regression, 
    partition data among clients (iid/non_iid), and select unlearning samples.
:src: [paper name] Federated Unlearning Compensation
"""

import os
from typing import Literal, Optional, Union

import pickle
import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms 
from torchvision.datasets import MNIST
from PIL import Image


BINARY_MNIST_NUM_CHANNEL = 1 
BINARY_MNIST_NUM_CLASS = 2

MNIST_MEAN = (0.1307, )
MNIST_STD = (0.3081, )

MNIST_ROOT = '../../data/mnist/'


class BinaryMNIST(Dataset): 
    def __init__(self, root: str=MNIST_ROOT, binary_class: tuple[int, int]=(3, 8), train=True, transform=None):
        """Binary MNIST dataset for binary classification

        :param binary_class: binary digit to classify, defaults to (3, 8)
        :param train: train dataset or test dataset, defaults to True
        :param transform: transformations to apply to the data
        """
        self.binary_class = binary_class
        self.train = train
        self.transform = transform

        original_mnist = MNIST(root=root, train=train, download=True, transform=None)
        data, targets = original_mnist.data, original_mnist.targets

        mask = (targets == binary_class[0]) | (targets == binary_class[1])
        data = data[mask]
        targets = targets[mask]

        # Map binary classes to 0 and 1
        targets = (targets == binary_class[1]).long()

        self.data = data
        self.targets = targets.to(torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform:
            img = self.transform(img)

        return img, label
    

def get_dataset(train=True, is_transform=True, root=MNIST_ROOT) -> Dataset:
    if is_transform: 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
    else: 
        transform = None

    return BinaryMNIST(root=root, train=train, transform=transform)


def get_unlearn_indices(data: Dataset, indices: list[int], unlearn_perc: Optional[float]=None, target_label: int=0, src_label: Union[int, list[int]]=None) -> list[int]: 
    if unlearn_perc is None: 
        unlearn_perc = np.random.uniform(0, 0.5)
    if src_label is not None:
        if isinstance(src_label, int):
            src_label = [src_label]
        candidate_indices = [i for i in indices if data.targets[i] in src_label]
    else:
        candidate_indices = [i for i in indices if data.targets[i] != target_label]
    
    num_unlearn = min(int(len(indices) * unlearn_perc), len(candidate_indices))
    unlearn_indices = np.random.choice(candidate_indices, num_unlearn, replace=False).tolist()
    return unlearn_indices 



def to_hwc(image):
    """ Convert image to HWC format.
    
    :param image: Input image, can be HWC, CHW, or HW (grayscale).
    :return: Image in HWC format.
    """
    if len(image.shape) == 2:  # HW (grayscale)
        return image[:, :, np.newaxis]  # -> HWC (H, W, 1)
    
    elif len(image.shape) == 3:
        # Check if CHW: (3, 32, 32) -> assume CHW
        if image.shape[0] == 3 or image.shape[0] == 1:  # Likely CHW
            return np.transpose(image, (1, 2, 0))  # CHW -> HWC
        else:  # Likely HWC
            return image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    

def to_chw(image): 
    """ Convert image to CHW format.
    
    :param image: Input image, can be HWC, CHW, or HW (grayscale).
    :return: Image in CHW format.
    """
    if len(image.shape) == 2:  # HW (grayscale)
        return image[np.newaxis, :, :]  # -> CHW (1, H, W)
    
    elif len(image.shape) == 3:
        # Check if HWC: (32, 32, 3) -> assume HWC
        if image.shape[2] == 3 or image.shape[2] == 1:  # Likely HWC
            return np.transpose(image, (2, 0, 1))  # HWC -> CHW
        else:  # Likely CHW
            return image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def to_hw(image):
    """ Convert image to HW format.
    
    :param image: Input image, can be HWC, CHW, or HW (grayscale).
    :return: Image in HW format.
    """
    if len(image.shape) == 2:  # Already HW
        return image
    
    elif len(image.shape) == 3:
        if image.shape[0] == 1 or image.shape[2] == 1:  # CHW or HWC with single channel
            return image[:, :, 0] if image.shape[2] == 1 else image[0, :, :]  # -> HW (H, W)
        else:
            raise ValueError(f"Unsupported image shape for HW conversion: {image.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def is_hwc(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[2] in [1, 3]


def is_chw(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[0] in [1, 3]


def is_hw(image: np.ndarray) -> bool:
    return len(image.shape) == 2


def add_backdoor_trigger(image, trigger_type: Literal['pixel', 'pattern'] = 'pattern', trigger_size=3, trigger_value=255, distance=2):
    """
    Add a backdoor trigger to an image.

    :param image: Input image
    :param trigger_type: Trigger type ('pixel' or 'pattern').
    :param trigger_size: Size of the backdoor trigger (default: 3x3).
    :param trigger_value: Value to fill the backdoor trigger (default: 255 for raw images).
    :param distance: Distance from bottom-right corner.
    :return: Image with backdoor trigger added.
    """
    # image_ = deepcopy(image)
    image_ = image

    image_ = to_hwc(image_)  # Ensure image is in HWC format
    h, w = image_.shape[:2]
    channel = image_.shape[2]

    if h < trigger_size + distance or w < trigger_size + distance:
        raise ValueError(f'Trigger size and distance exceed image boundaries: h = {h}, w = {w}, trigger_size = {trigger_size}, distance = {distance}')

    if trigger_type == 'pixel':
        image_[-trigger_size - distance: -distance, -trigger_size - distance: -distance, :] = trigger_value
    elif trigger_type == 'pattern':
        backdoor_pattern = np.array([[0, 0, 1],
                                     [0, 1, 0],
                                     [1, 0, 1]])
        pattern = backdoor_pattern
        for ch in range(channel):
            for i in range(trigger_size):
                for j in range(trigger_size):
                    if pattern[i][j] == 1:
                        image_[-trigger_size - distance + i, -trigger_size - distance + j, ch] = trigger_value
    else:
        raise ValueError("Invalid trigger type. Choose 'pixel' or 'pattern'.")

    if is_hw(image):
        image_ = to_hw(image_)
    elif is_chw(image):
        image_ = to_chw(image_)

    return image_


def client_indices_partition(data: Dataset, num_clients: int, partition: Literal['iid', 'non_iid']='iid', alpha: float=0.5, num_classes: int=BINARY_MNIST_NUM_CLASS, threshold=0.02) -> list[list[int]]: 
    num_sample = len(data)

    if partition == 'iid' or alpha >= 1e8: 
        indices = np.random.permutation(num_sample)
        return np.array_split(indices, num_clients)
    elif partition == 'non_iid':
        class_indices = [[] for _ in range(num_classes)]
        for idx, (_, label) in enumerate(data):
            class_indices[int(label)].append(idx)

        client_indices = [[] for _ in range(num_clients)]

        if alpha: 
            for c in range(num_classes):
                np.random.shuffle(class_indices[c])
                proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)
                num_class_samples = len(class_indices[c])
                proportions = (proportions * num_class_samples * (1 - threshold) + num_class_samples * threshold / num_clients).astype(int)

                print(f"Class {c} distribution among clients: {proportions}")

                start_idx = 0
                for i in range(num_clients):
                    end_idx = start_idx + proportions[i]
                    client_indices[i].extend(class_indices[c][start_idx:end_idx])
                    start_idx = end_idx 
        else: 
            num_clients_per_class = num_clients // num_classes

            # randomly choose num_clients_per_class clients for each class (make sure no client is chosen twice), and then assign all samples of that class to those clients averagely
            selected_clients = np.random.permutation(num_clients)
            for c in range(num_classes):
                np.random.shuffle(class_indices[c])
                selected_client = selected_clients[c * num_clients_per_class: min((c + 1) * num_clients_per_class, num_clients)]
                num_class_samples = len(class_indices[c])
                samples_per_client = num_class_samples // num_clients_per_class 

                start_idx = 0
                for i in selected_client:
                    end_idx = start_idx + samples_per_client
                    client_indices[i].extend(class_indices[c][start_idx:end_idx])
                    start_idx = end_idx 

        return client_indices
    else: 
        raise ValueError(f"Unsupported partition type: {partition}, expected 'iid' or 'non_iid'")


def load_dataloader(args): 
    train_dataset = get_dataset(train=True, is_transform=True)
    test_dataset = get_dataset(train=False, is_transform=True)
    backdoor_test_dataset = get_dataset(train=False, is_transform=True)

    # For iid partition (or alpha >= 1e8) the alpha suffix is omitted from the
    # index filename, matching the naming convention of the existing pkl files.
    is_iid = args.partition == 'iid' or args.alpha >= 1e8
    if is_iid:
        info = (
            args.dataset, args.seed,
            args.unlearn_select, args.target_label, args.src_label,
            args.unlearn_perc, args.unlearn_client_perc,
            args.num_clients, args.partition,
        )
    else:
        info = (
            args.dataset, args.seed,
            args.unlearn_select, args.target_label, args.src_label,
            args.unlearn_perc, args.unlearn_client_perc,
            args.num_clients, args.partition, args.alpha,
        )

    save_pth = f"cache/client_indices/{'_'.join(map(str, info))}.pkl"

    # check if the client indices and unlearn indices have been saved, if yes, load them, otherwise, generate new ones and save them
    if os.path.exists(save_pth):
        with open(save_pth, 'rb') as f:
            saved_data = pickle.load(f)
            client_indices = saved_data.get('client_indices')
            client_unlearn_indices = saved_data.get('unlearn_indices')
        print(f">>> Loaded client indices and unlearn indices from {save_pth}")
    else: 
        client_indices = client_indices_partition(train_dataset, num_clients=args.num_clients, partition=args.partition, alpha=args.alpha) 
        client_unlearn_indices = [get_unlearn_indices(train_dataset, indices, unlearn_perc=args.unlearn_perc, target_label=args.target_label, src_label=args.src_label) for indices in client_indices]
        if not os.path.exists(os.path.dirname(save_pth)): 
            os.makedirs(os.path.dirname(save_pth))
        with open(save_pth, 'wb') as f:
            pickle.dump({
                'client_indices': client_indices,
                'unlearn_indices': client_unlearn_indices
            }, f)
        print(f">>> Saved client indices and unlearn indices to {save_pth}")

    unlearn_indices = [idx for indices in client_unlearn_indices for idx in indices]

    # add backdoor trigger to the train_dataset[unlearn_indices] and backdoor_test_dataset
    for idx in unlearn_indices: 
        train_dataset.data[idx] = add_backdoor_trigger(train_dataset.data[idx])
        train_dataset.targets[idx] = args.target_label  # change label to target_label

    test_unlearn_indices = [idx for idx in range(len(test_dataset)) if test_dataset.targets[idx] == args.src_label]
    backdoor_test_dataset.data = test_dataset.data[test_unlearn_indices]
    for idx in range(len(backdoor_test_dataset)): 
        backdoor_test_dataset.data[idx] = add_backdoor_trigger(backdoor_test_dataset.data[idx])
        backdoor_test_dataset.targets[idx] = args.target_label  # change label to target_label 

    train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=args.batch_size, shuffle=True) for indices in client_indices]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loaders, test_loader, backdoor_test_loader, client_indices, client_unlearn_indices, train_dataset


def dataloader_to_numpy(dataloader):
    X_list = []
    y_list = []
    for inputs, targets in dataloader:
        # flat the inputs
        inputs = inputs.view(inputs.size(0), -1)
        X_list.append(inputs.numpy())
        # need to convert {0, 1} to {-1, +1} for logistic regression
        y_list.append((2 * targets.numpy()) - 1)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X.astype(np.float64), y.astype(np.float64)