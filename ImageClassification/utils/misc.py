"""
:file: misc.py
:date: 2025-07-10
:description: Other tools and utilities 
"""


import random 
import pickle
import os
from typing import Dict, List, Literal, Tuple, Union, Optional, Any 

import torch 
import torch.nn as nn
import numpy as np


def set_cuda(device_idx: int) -> torch.device: 
    """ Set the CUDA device to be used. 

    :param device_idx: Index of the CUDA device to be used.
    :return: The device object representing the selected CUDA device.
    """
    device = torch.device('cuda:{}'.format(device_idx) if torch.cuda.is_available() and device_idx != -1 else 'cpu') 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    return device


def set_seed(seed: int) -> int:
    """ Set the random seed for reproducibility. 

    :param seed: The seed value to set.
    :return: The seed value that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def save_model(model: Union[nn.Module, dict], path: str, folder: str = "cache/models") -> None:
    """ Save the model to a file.

    :param model: The model to save, can be a nn.Module or a state_dict.
    :param path: The file path where the model will be saved.
    """
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), f"{folder}/{path}.pth")
    elif isinstance(model, dict):
        torch.save(model, f"{folder}/{path}.pth")
    else:
        raise ValueError("Model must be nn.Module or dict")
    

def load_model(model: nn.Module, path: str, device: torch.device, folder: str="cache/models") -> nn.Module:
    """ Load the model from a file.

    :param model: The model to load, can be a nn.Module or a state_dict.
    :param path: The file path from which the model will be loaded.
    :param device: The device on which the model will be loaded.
    :return: The loaded model.
    """
    load_info = torch.load(f"{folder}/{path}.pth", map_location=device)
    if isinstance(load_info, nn.Module):
        model.load_state_dict(load_info.state_dict())
        return model
    elif isinstance(load_info, dict):
        model.load_state_dict(load_info)
        return model
    else:
        raise ValueError(f"Loaded model {path} is not nn.Module or dict")


def save_checkpoint(state: Dict[str, Any], path: str, folder: str = "cache/checkpoints") -> None:
    """Save the training state to a checkpoint file.

    :param state: information to save, typically includes model state, optimizer state, epoch, etc.
    :param path: the file path where the checkpoint will be saved.
    :param folder: saving folder, defaults to "cache/checkpoints"
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, f"{folder}/{path}.pth")
    # print(f">>> Checkpoint saved to {folder}/{path}.pth")


def load_checkpoint(path: str, folder: str = "cache/checkpoints") -> Dict[str, Any]:
    """Load the training state from a checkpoint file.

    :param path: the file path from which the checkpoint will be loaded.
    :param folder: loading folder, defaults to "cache/checkpoints"
    :return: the loaded state dictionary.
    """
    if not os.path.exists(f"{folder}/{path}.pth"):
        raise FileNotFoundError(f"Checkpoint {folder}/{path}.pth not found")
    
    state = torch.load(f"{folder}/{path}.pth", map_location='cpu')
    # print(f">>> Checkpoint loaded from {folder}/{path}.pth")
    return state


def load_checkpoint_to_model(model: nn.Module, path: str, folder: str = "cache/checkpoints") -> nn.Module: 
    """ Load the model state from a checkpoint file.

    :param model: The model to load the state into.
    :param path: The file path from which the checkpoint will be loaded.
    :param folder: loading folder, defaults to "cache/checkpoints"
    :param device: The device on which the model will be loaded.
    :return: The model with the loaded state.
    """
    if not os.path.exists(f"{folder}/{path}.pth"):
        raise FileNotFoundError(f"Checkpoint {folder}/{path}.pth not found")
    
    state = torch.load(f"{folder}/{path}.pth", map_location='cpu')
    nn.utils.vector_to_parameters(state['params'], model.parameters())
    
    print(f">>> Model state loaded from {folder}/{path}.pth")
    return model


def get_model_update(local_model: Union[nn.Module, Dict[str, torch.Tensor]], global_model: Union[nn.Module, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]: 
    """ Get the model update from local model to global model.
    :param local_model: The local model from which the update is computed.
    :param global_model: The global model to which the update is applied.
    :return: A dictionary containing the model update.
    """
    if isinstance(local_model, nn.Module):
        local_model = local_model.state_dict()
    if isinstance(global_model, nn.Module):
        global_model = global_model.state_dict()

    return {
        name: local_model[name] - global_model[name] 
        for name in local_model.keys() if name in global_model
    }


def save_client_data_indices(indices: List[List[int]], unlearn_indices: List[List[int]], args, folder="cache/indices") -> None: 
    """Save client data indices and unlearning indices to a file.

    :param indices: The client data indices to save.
    :param unlearn_indices: The unlearning indices to save.
    :param args: Additional arguments, defaults to None.
    """
    info = (
        args.dataset, args.seed, 
        args.unlearn_select, args.target_label, args.src_label,
        args.unlearn_perc, args.unlearn_client_perc, 
        args.num_clients, args.partition, args.alpha, 
    )

    if not os.path.exists(folder):
        os.makedirs(folder)

    save_pth = f"{folder}/{'_'.join(map(str, info))}.pkl"

    with open(save_pth, 'wb') as f: 
        pickle.dump({
            'client_indices': indices, 
            'unlearn_indices': unlearn_indices
        }, f)
    
    print(f">>> Client data indices saved to {save_pth}")


def load_client_data_indices(args) -> Optional[Tuple[List[List[int]], List[List[int]]]]: 
    """Load client data indices and unlearning indices from a file.

    :param args: Additional arguments, defaults to None.
    :return: A tuple containing the client indices and unlearning indices, or None if the file does not exist.
    """
    info = (
        args.dataset, args.seed, 
        args.unlearn_select, args.target_label, args.src_label,
        args.unlearn_perc, args.unlearn_client_perc, 
        args.num_clients, args.partition, args.alpha, 
    )
    load_pth = f"cache/indices/{'_'.join(map(str, info))}.pkl"

    if not os.path.exists(load_pth):
        print(f">>> Client data indices file not found: {load_pth}")
        return None, None

    with open(load_pth, 'rb') as f:
        data = pickle.load(f)
        client_indices = data.get('client_indices')
        unlearn_indices = data.get('unlearn_indices')
        if client_indices is None or unlearn_indices is None:
            print(f">>> Client data indices or unlearn indices not found in {load_pth}")
            return None, None
        else:
            print(f">>> Client data indices loaded from {load_pth}")
            return client_indices, unlearn_indices
        

def get_distance(model1: nn.Module, model2: nn.Module, mode: Literal['l2', 'l1', 'linf', 'cosine'] = 'l2') -> float:
    """Compute distance between two models' parameters.

    :param model1: First model to compare.
    :param model2: Second model to compare.
    :param mode: Distance mode, options: "l2", "l1", "linf", "cosine".
    :return: Distance value.
    """
    with torch.no_grad():
        v1 = nn.utils.parameters_to_vector(model1.parameters())
        v2 = nn.utils.parameters_to_vector(model2.parameters())
        diff = v1 - v2

        if mode == "l2":
            return torch.norm(diff, p=2).item()
        elif mode == "l1":
            return torch.norm(diff, p=1).item()
        elif mode == "linf":
            return torch.norm(diff, p=float('inf')).item()
        elif mode == "cosine":
            cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            return 1 - cos
        else:
            raise ValueError(f"Unsupported mode: {mode}")

