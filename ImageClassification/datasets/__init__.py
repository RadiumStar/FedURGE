"""
:file: __init__.py
:date: 2025-07-10
:description: data loading module
"""

from . import svhn, cifar10, cifar100, tiny_imagenet
from .processor import *

ROOT_FOLDER = '../../data/'  # default root folder for datasets


def get_num_class(dataset_name: str) -> int:
    """Get the number of classes for a given dataset.

    :param dataset_name: Name of the dataset (e.g., 'mnist')
    :return: Number of classes in the dataset
    """
    try: 
        return eval(f"{dataset_name.lower()}.{dataset_name.upper()}_NUM_CLASS")
    except AttributeError:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_num_channel(dataset_name: str) -> int:
    """Get the number of channels for a given dataset.

    :param dataset_name: Name of the dataset (e.g., 'mnist')
    :return: Number of channels in the dataset
    """
    try:
        return eval(f"{dataset_name.lower()}.{dataset_name.upper()}_NUM_CHANNEL")
    except AttributeError:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    

def get_dataset(dataset_name: str, train=True, is_transform=True, root_folder=ROOT_FOLDER) -> Dataset: 
    """Get the dataset from the dataset name.

    :param dataset_name: Name of the dataset (e.g., 'mnist')
    :param train: Whether to load the training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :param root_folder: Root folder for the dataset, defaults to ROOT_FOLDER
    :return: Dataset object
    """
    data = eval(f"{dataset_name.lower()}.get_dataset(train, is_transform, root_folder)")
    data.num_channel = get_num_channel(dataset_name)
    data.num_class = get_num_class(dataset_name)
    return data 


def get_backdoor_dataset(base_data: Dataset, backdoor_indices: List[int] = None, backdoor_ratio: float = None, target_label: int = 0, src_label: Union[int, List[int]] = None, only_backdoor: Optional[bool] = False, trigger_type: Literal['pixel', 'pattern'] = 'pattern', trigger_size: int = 3, trigger_value: int = 255, distance: int = 2) -> BackdoorDataset: 
    """get backdoor dataset from the base dataset.

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
    :return: BackdoorDataset object
    """
    if backdoor_indices is None and backdoor_ratio is None:
        raise ValueError("Either backdoor_indices or backdoor_ratio must be provided")
    
    return BackdoorDataset(base_data, backdoor_indices, backdoor_ratio, target_label, src_label, only_backdoor, trigger_type, trigger_size, trigger_value, distance)
    

def get_client_indices(data: Dataset, num_client=10, partition: Literal['iid', 'non_iid'] = 'iid', alpha: Optional[float] = 0.5, num_class: int = 10) -> List[List[int]]:
    """Get the client indices from the dataset.

    :param dataset: Dataset to be split
    :param num_client: Number of clients, defaults to 10
    :param partition: Partition type ('iid' or 'non_iid'), defaults to 'iid'
    :param alpha: Alpha parameter for non_iid partition (Dirichlet distribution), defaults to 0.5
    :return: List of client datasets
    """
    return split_dataset(data, num_client, partition, alpha, num_class)


def get_unlearn_indices(base_data: Dataset, indices: List[int] = None, unlearn_select: Literal['class', 'random', 'backdoor'] = 'random', unlearn_perc: Optional[float] = None, target_label: Union[int, List[int]] = None, src_label: Union[int, List[int]] = None) -> List[int]:
    """Get the indices for unlearning.

    :param data: Dataset to be processed
    :param indices: Indices of the dataset to be processed, defaults to None (all indices)
    :param unlearn_select: Scenarios to select unlearning indices ('class' or 'random'), defaults to 'random'
    :param unlearn_perc: Percentage of data to unlearn, defaults to None
    :param target_label: Target label for class unlearning, defaults to None; for backdoor unlearning, it can only be a single label (`int`)
    :param src_label: Source label for backdoor unlearning, defaults to None
    :return: List of indices for unlearning
    """
    data = base_data if indices is None else Subset(base_data, indices)
    if unlearn_select == 'class':
        return get_class_indices(data, target_label, unlearn_perc)
    elif unlearn_select == 'random':
        return get_random_indices(data, unlearn_perc)
    elif unlearn_select == 'backdoor':
        return get_backdoor_indices(data, target_label, src_label, unlearn_perc)
    else:
        raise ValueError(f"Unsupported unlearn selection method: {unlearn_select}")