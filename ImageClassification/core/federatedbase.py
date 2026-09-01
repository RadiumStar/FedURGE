"""
:file: client.py
:date: 2025-07-11
:description: Client Base Class for Federated Learning and Unlearning 
"""

from typing import Any, Optional, Union, Literal, List, Dict 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from utils import Communicator


class Client: 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), args=None): 
        """Client Base Class for Federated Learning and Unlearning

        :param global_dataset: global dataset for the client
        :param data_indices: indices of the local dataset for the client
        :param local_model: local model with the same architecture as the global model
        :param client_id: client index
        :param comm: Communicator for communication between client and server
        :param args: other arguments, defaults to None
        """

        self.global_dataset = global_dataset
        self.local_dataset = Subset(global_dataset, data_indices)
        self.data_indices = data_indices
        self.local_model = local_model 
        self.local_model.cuda()
        self.client_id = client_id 

        self.args = args

        self.set_optimizer(args=args)
            
        self.comm = comm
        
        self.criterion = getattr(self.args, 'criterion', 'CrossEntropyLoss')
        self.criterion = eval(f"nn.{self.criterion}")() if isinstance(self.criterion, str) else self.criterion

    def set_model(self, global_model: Union[nn.Module, str, dict]) -> None:
        """Set global model to local model

        :param global_model: global model or path to the model file or state_dict
        """
        if self.local_model is None: 
            raise ValueError("local_model is not initialized") 

        if isinstance(global_model, nn.Module):
            self.local_model.load_state_dict(global_model.state_dict())
        elif isinstance(global_model, str):
            self.local_model.load_state_dict(torch.load(global_model))
        elif isinstance(global_model, dict):
            self.local_model.load_state_dict(global_model)
        else:
            raise ValueError("global_model must be nn.Module, str or dict")
        
    def set_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> None:
        """Set dataloader for local dataset

        :param batch_size: batch size for the dataloader, defaults to 64
        :param shuffle: whether to shuffle the dataset, defaults to True
        """
        self.train_loader = DataLoader(
            self.local_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=2, 
            pin_memory=True, 
            drop_last=True
        )

    def set_optimizer(self, args=None, optimizer: Literal['SGD', 'Adam'] = 'SGD', lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        """Reset optimizer for the global model

        :param args: [WARNING] useless arguments, should be removed in the future, defaults to None, we use `self.args` instead
        :param optimizer: type of optimizer, defaults to 'SGD'
        :param lr: learning rate, defaults to 0.01
        :param momentum: momentum for SGD, defaults to 0.0
        :param weight_decay: weight decay for optimizer, defaults to 0.0
        """
        self.opt = getattr(self.args, 'optimizer', optimizer)
        if self.opt == 'SGD':
            self.opt = torch.optim.SGD(
                self.local_model.parameters(), 
                lr=getattr(self.args, 'lr', lr), 
                momentum=getattr(self.args, 'momentum', momentum), 
                weight_decay=getattr(self.args, 'weight_decay', weight_decay)
            )
        elif self.opt == 'Adam':
            self.opt = torch.optim.Adam(
                self.local_model.parameters(), 
                lr=getattr(self.args, 'lr', lr), 
                betas=(getattr(self.args, 'beta1', 0.9), getattr(self.args, 'beta2', 0.999)),
                weight_decay=getattr(self.args, 'weight_decay', weight_decay)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt}")

    @torch.no_grad()
    def communicate(self, global_model: nn.Module) -> Dict[str, Any]:
        """Communicate local model parameters to the server

        :param global_model: global model to which the local model is compared
        :return: local model state_dict
        """
        update: Dict[str, torch.Tensor] = {}
        local_params = dict(self.local_model.named_parameters()) 
        global_params = dict(global_model.named_parameters()) 

        # divide_lr = 1 if self.args.communicator in ('Communicator', 'EF14') else self.opt.param_groups[0]['lr']

        divide_lr = 1

        for name in local_params.keys(): 
            update[name] = (local_params[name] - global_params[name]) / divide_lr
        
        update = self.comm(update)

        for name in local_params.keys(): 
            update[name] = update[name] * divide_lr + global_params[name]

        local_state_dict = self.local_model.state_dict()
        for name in update.keys():
            local_state_dict[name] = update[name]
        
        return local_state_dict
    
    def train(self, loader: Optional[DataLoader] = None) -> None: 
        raise NotImplementedError("train method must be implemented in the subclass")
    
    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Local unlearning method to be implemented in the subclass

        :param loader: dataloader for training, defaults to None
        """
        raise NotImplementedError("local_unlearning method must be implemented in the subclass")
    
    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method to be implemented in the subclass

        :param loader: dataloader for training, defaults to None
        """
        raise NotImplementedError("local_refinement method must be implemented in the subclass")
    
    def get_communication_cost(self, unit: Optional[Literal['B', 'KB', 'MB', 'GB']] = 'MB') -> float:
        """Get communication cost in selected unit

        :param unit: unit of communication cost, defaults to 'MB'
        :return: communication cost in selected unit
        """
        return self.comm.get_communication_cost(unit=unit)

    def __repr__(self):
        return f"Client(client_id={self.client_id}, model={self.local_model.__class__.__name__})"
    

class Server: 
    def __init__(self, global_model: nn.Module, args=None):
        """Server Base Class for Federated Learning and Unlearning

        :param global_model: global model for the server
        :param args: other arguments, defaults to None
        """
        self.global_model = global_model
        self.global_model.cuda()

        self.args = args
        
        self.criterion = getattr(self.args, 'criterion', 'CrossEntropyLoss')
        self.criterion = eval(f"nn.{self.criterion}")() if isinstance(self.criterion, str) else self.criterion
        
    def set_model(self, model: Union[nn.Module, str, dict]) -> None:
        """Set global model to server

        :param model: global model or path to the model file or state_dict
        """
        if isinstance(model, nn.Module):
            self.global_model.load_state_dict(model.state_dict())
        elif isinstance(model, str):
            self.global_model.load_state_dict(torch.load(model))
        elif isinstance(model, dict):
            self.global_model.load_state_dict(model)
        else:
            raise ValueError("model must be nn.Module, str or dict")

    @torch.no_grad()
    def aggregate(self, clients: List[Client]) -> None: 
        """Aggregate client models

        :param clients: list of Client
        """
        uploads = [client.communicate(self.global_model) for client in clients]
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.mean(
                torch.stack([upload[key].float() for upload in uploads]), dim=0
            )
        self.global_model.load_state_dict(global_state_dict)

    def evaluate(self, test_loader: DataLoader, criterion=None, evaluated_model=None) -> float:
        """Evaluate global model

        :param test_loader: dataloader for evaluation
        :param criterion: loss function for evaluation, defaults to None
        :param evaluated_model: model to be evaluated, defaults to None (use global model)
        :return: accuracy of the global model
        """ 

        if evaluated_model is not None:
            model = evaluated_model
        else:
            model = self.global_model

        # if the dataset is binary classification, use sigmoid activation
        if self.args.num_classes == 2: 
            model.eval()
            correct = 0
            total = 0
            total_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    x = x.to(torch.float32)
                    logits = model(x)
                    if criterion: total_loss += criterion(logits, y).item()
                    pred = (torch.sigmoid(logits) >= 0.5).long()
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            total_loss /= len(test_loader)
            acc = correct / total
            if criterion: 
                return acc, total_loss
            else: 
                return acc
        # if the dataset is multi-class classification, use softmax activation
        else: 
            model.eval()
            correct = 0
            total = 0
            total_loss = 0.0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    if criterion: total_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            total_loss /= len(test_loader)
            accuracy = correct / total if total > 0 else 0.0
            if criterion: 
                return accuracy, total_loss
            else: 
                return accuracy
    
    def __call__(self, clients: List[Client], args=None) -> nn.Module:
        """Run Federated Learning with the server

        :param clients: List of client instances
        :param args: Other arguments, defaults to None
        :return: Updated global model
        """
        raise NotImplementedError("federated_learning method must be implemented in the subclass")