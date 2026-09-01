"""
:file: processor.py
:date: 2025-07-10
:description: data processing utilities
"""


from typing import Union, Optional, Literal, List, Dict 
from copy import deepcopy
 
import matplotlib.pyplot as plt 
import PIL
import numpy as np 
import torch
from torch.utils.data import Dataset, Subset, DataLoader


BACKDOOR_PATTERNS = {
    1: np.array([[1]]),
    3: np.array([[0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 1]]),
    4: (np.arange(16).reshape(4, 4) % 2),
    5: np.ones((5, 5)), 
    6: np.ones((6, 6)),
    8: np.ones((8, 8)), 
    10: np.ones((10, 10)), 
    11: np.ones((11, 11))
}

 
def visualize_imgs(images, num_imgs=5, save_path=None):
    plt.figure(figsize=(num_imgs * 2, 2))
    for i in range(num_imgs):
        img = images[i][0]  # get the image tensor
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # CHW to HWC
        plt.subplot(1, num_imgs, i + 1)
        plt.imshow(img)
        plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def split_dataset_iid(num_sample: int, num_client: int) -> List[List[int]]:
    """
    Split dataset indices into iid subsets for clients.

    :param num_sample: Total number of samples in the dataset
    :param num_client: Number of clients
    :return: List of lists containing indices for each client
    """
    indices = list(range(num_sample))
    np.random.shuffle(indices)
    num_per_client = num_sample // num_client
    return [indices[i * num_per_client:(i + 1) * num_per_client] for i in range(num_client)]


def split_dataset_non_iid(targets: List[int], num_client: int, alpha: float = 0.5, num_class: int = 10) -> List[List[int]]: 
    """
    Split dataset indices into non-iid subsets for clients.

    :param targets: List of class labels for each sample
    :param num_client: Number of clients
    :param alpha: Parameter for Dirichlet distribution
    :param num_class: Number of classes in the dataset
    :return: List of lists containing indices for each client
    """
    class_indices = [[] for _ in range(num_class)]
    
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_client)]
    
    for class_idx in class_indices:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet([alpha] * num_client)
        proportions = (len(class_idx) * proportions).astype(int)
        split_indices = np.split(class_idx, np.cumsum(proportions)[:-1])
        
        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx.tolist())

    return client_indices


def split_dataset(data: Dataset, num_client: int, partition: Literal['iid', 'non_iid'] = 'iid', alpha: Optional[float] = 0.5, num_class: int = 10) -> List[List[int]]:
    """
    Split dataset into subset indices for clients.

    :param data: Dataset to be split
    :param num_client: Number of clients
    :param partition: Partition type ('iid' or 'non_iid')
    :param alpha: Parameter for non_iid partitioning
    :param num_classes: Number of classes in the dataset
    :return: List of subsets for each client
    """
    num_sample = len(data) 

    if partition == 'iid': 
        client_indices = split_dataset_iid(num_sample, num_client)
    elif partition == 'non_iid': 
        if hasattr(data, 'is_backdoor_poisoned') and data.is_backdoor_poisoned:
            client_indices = split_dataset_non_iid(data.base_data.targets, num_client, alpha, num_class)
        else: 
            client_indices = split_dataset_non_iid(data.targets, num_client, alpha, num_class)
    else: 
        raise ValueError("Invalid partition type. Choose 'iid' or 'non_iid'.")
    
    return client_indices


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
        backdoor_pattern = BACKDOOR_PATTERNS[trigger_size]
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


def get_backdoor_indices(data: Union[Dataset, Subset], target_label: int = 0, src_label: Union[int, List[int]] = None, unlearn_perc: Optional[float] = None) -> List[int]:
    """get indices for backdoor unlearning

    :param data: data to be processed
    :param target_label: target label for backdoor unlearning 
    :param src_label: source label(s) for backdoor unlearning, if None, it will backdoor all samples except target_label
    :param unlearn_perc: percentage of samples to be unlearned, if None, it will randomly select a percentage of samples between 0 and 0.5 
    :return: list of indices to be unlearned
    """
    if hasattr(data, 'backdoor_indices'):
        return data.backdoor_indices
    elif isinstance(data, Subset):
        dataset = data.dataset
        indices = data.indices
    elif isinstance(data, Dataset):
        dataset = data
        indices = list(range(len(dataset)))
    else:
        raise ValueError("Invalid dataset type. Must be Dataset or Subset.")
    
    if unlearn_perc is None:
        unlearn_perc = np.random.uniform(0, 0.5)

    # Only select indices whose label is in src_label
    if src_label is not None:
        if isinstance(src_label, int):
            src_label = [src_label]
        candidate_indices = [i for i, idx in enumerate(indices) if dataset.targets[idx] in src_label]
    # Otherwise, select indices whose label is not target_label
    else:
        candidate_indices = [i for i, idx in enumerate(indices) if dataset.targets[idx] != target_label]

    num_unlearn = min(int(len(indices) * unlearn_perc), len(candidate_indices))

    local_indices = np.random.choice(candidate_indices, num_unlearn, replace=False)
    unlearn_indices = [indices[i] for i in local_indices]
    return unlearn_indices


def get_class_indices(data: Union[Dataset, Subset], target_label: Union[int, List[int]] = None, unlearn_perc: Optional[float] = None) -> List[int]: 
    """generate class indices for class unlearning

    :param data: dataset to be processed
    :param target_label: target class(es) to be unlearned, if None, it will randomly select a class or `unlearn_perc` is provided, it will randomly select a percentage of samples
    :param unlearn_perc: class unlearned percentage, defaults to None
    :return: list of indices to be unlearned
    """
    if isinstance(data, Subset):
        dataset = data.dataset
        indices = data.indices
    elif isinstance(data, Dataset): 
        dataset = data 
        indices = list(range(len(dataset)))
    else: 
        raise ValueError("Invalid dataset type. Must be Dataset or Subset.")

    # local_data = dataset.data 
    local_targets = dataset.targets

    if target_label is None: 
        num_unlearn = int(len(indices) * unlearn_perc) or 1 if unlearn_perc else 1
        local_indices = np.random.choice(len(indices), num_unlearn, replace=False)
        target_label = [local_targets[indices[i]] for i in local_indices]
    elif isinstance(target_label, int):
        target_label = [target_label]
    
    print(f"Target label: {target_label}")
    local_indices = [i for i, target in enumerate(local_targets) if target in target_label]
    unlearn_indices = [i for i in local_indices] # global indices for data provided
    return unlearn_indices


def get_random_indices(data: Dataset, unlearn_perc: Optional[float] = None) -> List[int]: 
    """generate random indices for unlearning

    :param data: dataset to be processed
    :param unlearn_perc: unlearn percentage, if None, it will randomly select a percentage of samples between 0 and 0.5
    :return: list of indices to be unlearned
    """
    if isinstance(data, Subset):
        dataset = data.dataset
        indices = data.indices
    elif isinstance(data, Dataset):
        dataset = data
        indices = list(range(len(dataset)))
    else:
        raise ValueError("Invalid dataset type. Must be Dataset or Subset.")
    
    if unlearn_perc is None:
        unlearn_perc = np.random.uniform(0, 0.5)
    num_unlearn = int(len(indices) * unlearn_perc)
    # print(f"Unlearn percentage: {unlearn_perc}, Number of samples to unlearn: {num_unlearn}")

    local_indices = np.random.choice(len(indices), num_unlearn, replace=False)
    unlearn_indices = [i for i in local_indices]
    return unlearn_indices


def get_data_loader(data: Dataset, batch_size: int = 64, shuffle: bool = True, indices: Optional[List[int]] = None, num_workers: int = 0):
    """
    Create a DataLoader for the dataset.

    :param data: Dataset to be loaded
    :param batch_size: Size of each batch
    :param shuffle: Whether to shuffle the dataset
    :param indices: If provided, only load the subset of the dataset with these indices
    :param num_workers: Number of subprocesses to use for data loading
    :return: DataLoader for the dataset
    """
    if indices is not None:
        if len(indices) == 0:
            return DataLoader([])
        else:
            return DataLoader(Subset(data, indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else: 
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    

class BackdoorDataset(Dataset): 
    def __init__(self, base_data: Dataset, backdoor_indices: List[int] = None, backdoor_ratio: float = None, target_label: int = 0, src_label: Union[int, List[int]] = None, only_backdoor: Optional[bool] = False, trigger_type: Literal['pixel', 'pattern'] = 'pattern', trigger_size: int = 3, trigger_value: int = 255, distance: int = 2): 
        """backdoor dataset for unlearning

        :param base_data: base clean dataset to be poisoned
        :param backdoor_indices: indices of samples to be poisoned, if None, it will depend on `backdoor_ratio`
        :param backdoor_ratio: ratio of samples to be poisoned, if 1.0, all samples will be poisoned
        :param target_label: backdoor attack target label
        :param src_label: source label(s) to be backdoored, if None, it will backdoor all samples except target_label
        :param only_backdoor: dataset will only contain backdoor samples, defaults to False
        :param trigger_type: backdoor trigger type (`'pixel'` or `'pattern'`), defaults to `'pattern'`
        :param trigger_size: backdoor trigger size, defaults to 3
        :param trigger_value: backdoor trigger value, defaults to 255
        :param distance: distance from image border, defaults to 2
        """
        self.base_dataset = base_data
        self.num_channel = base_data.num_channel
        self.num_class = base_data.num_class
        self.is_backdoor_poisoned = backdoor_ratio is not None and backdoor_ratio > 0.0

        self.transform = getattr(base_data, 'transform', None) 
        self.dataset = deepcopy(base_data)
        self.indices = base_data.indices if hasattr(base_data, 'indices') else list(range(len(base_data)))
        
        if backdoor_indices is not None:
            self.backdoor_indices = backdoor_indices
        else: 
            self.backdoor_indices = get_backdoor_indices(
                self.dataset, target_label=target_label, src_label=src_label, unlearn_perc=backdoor_ratio
            )

        self.backdoor_indices_set = set(self.backdoor_indices)

        for idx in self.backdoor_indices: 
            self.dataset.data[idx] = add_backdoor_trigger(
                self.dataset.data[idx], 
                trigger_type=trigger_type, 
                trigger_size=trigger_size, 
                trigger_value=trigger_value, 
                distance=distance
            )
            self.dataset.targets[idx] = target_label

        self.only_backdoor = only_backdoor or backdoor_ratio == 1.0

    def __len__(self):
        if self.only_backdoor:
            return len(self.backdoor_indices)
        return len(self.indices)
    
    def __getitem__(self, i):
        if self.only_backdoor:
            idx = self.backdoor_indices[i]
        else:
            idx = self.indices[i]
        
        x, y = self.dataset.data[idx], self.dataset.targets[idx]
        
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 3:
                x = PIL.Image.fromarray(x.numpy(), mode="RGB")
            else:
                x = PIL.Image.fromarray(x.numpy(), mode="L")
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = PIL.Image.fromarray(x, mode="RGB")
            else:  
                x = PIL.Image.fromarray(x, mode="L")

        if self.transform:
            x = self.transform(x)
        
        return x, y

    def is_backdoor(self, i) -> bool:
        """
        Check if the i-th sample (relative to this dataset) is a backdoor sample.
        """
        if self.only_backdoor:
            return True
        return self.indices[i] in self.backdoor_indices_set

        