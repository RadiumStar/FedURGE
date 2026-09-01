""" 
:file: fedmemprune.py
:date: 2026-06-18
:description: Federated Unlearning with Memory Pruning (FedMemPrune) Algorithm
:src: [Rethinking Federated Unlearning via the Lens of Memorization (KDD 2026)](https://dl.acm.org/doi/abs/10.1145/3770855.3817785); [Github Repo](https://github.com/JWei1999/Rethinking-Federated-Unlearning-via-the-Lens-of-Memorization)
"""
 
import random
import time
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np 
from torch.cuda.amp import autocast, GradScaler

from .federatedbase import Client, Server
from utils import Communicator, save_checkpoint, save_model


class ClientFedMemPrune(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Random Labeling with Memory Pruning (FedMemPrune) Algorithm
        
        :param global_dataset: Global dataset for the client
        :param data_indices: Indices of the local dataset for the client
        :param local_model: Local model
        :param client_id: Client index, defaults to 0
        :param comm: Communicator for communication between client and server, defaults to Communicator()
        :param args: Other arguments
        """
        super().__init__(global_dataset, data_indices, local_model, client_id, comm, args=args)

        self.set_dataloader(
            batch_size=getattr(self.args, 'batch_size', len(self.data_indices)), 
            shuffle=getattr(self.args, 'shuffle', True),
            unlearning_indices=unlearning_indices
        )

        self.cos_threshold = getattr(self.args, 'cos_threshold', -1)


    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None: 
        """ only need remaining dataset """
        super().set_dataloader(batch_size, shuffle)

        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_refine = bool(self.remaining_indices) 
        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.join_refine else None

    def local_unlearning(self, loader: Optional[DataLoader] = None, epoch: int = 0) -> None:
        """Local unlearning method for FedMemPrune algorithm

        :param loader: Dataloader for unlearning, defaults to None
        :param epoch: Current epoch, defaults to 0
        """
        return self.local_refinement(loader)


    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedMemPrune algorithm

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        if loader is None:
            loader = self.remaining_loader
        
        self.local_model.train()
        
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            self.opt.zero_grad()
            output = self.local_model(data)
            loss = self.criterion(output, target)
            loss.backward() 

        self.opt.step()


class ServerFedMemPrune(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for Federated Random Labeling with Memory Pruning (FedMemPrune) Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args=args)
        drop_ratios_dict = {
            "iid": {
                "mnist": 0.3, "fashion_mnist": 0.3, 
                "cifar10": 0.3, "svhn": 0.3, 
                "cifar100": 0.2,"tiny_imagenet": 0.2,
            }, 
            "non_iid": {
                "mnist": 0.4, "fashion_mnist": 0.4, 
                "cifar10": 0.4, "svhn": 0.4, 
                "cifar100": 0.2,"tiny_imagenet": 0.2,
            }
        }
        self.drop_ratio = drop_ratios_dict.get(getattr(self.args, 'partition', 'iid'), {}).get(getattr(self.args, 'dataset', 'mnist'), 0.5)
        print(f">>> Using drop ratio: {self.drop_ratio:.2f} for dataset {getattr(self.args, 'dataset', 'mnist')} with partition {getattr(self.args, 'partition', 'iid')}")

    def get_memorization_mask(self, loader: DataLoader) -> Dict[str, Any]: 
        """Locate memorization parameters

        :param loader: remaining loader for all remaining clients
        :return: mask
        """
        self.global_model.eval()
        self.global_model.zero_grad()

        data, target = next(iter(loader))
        data, target = data.cuda(), target.cuda()
        output = self.global_model(data)
        loss = self.criterion(output, target)
        loss.backward() 
        
        mask = {}
        # 1. Collect all absolute gradients and their corresponding parameter names/shapes
        grad_items = [] # (name, flat_abs_grad, shape)
        total_params = 0
        for name, param in self.global_model.named_parameters():
            if param.grad is not None:
                flat_abs_grad = param.grad.abs().view(-1)
                grad_items.append((name, flat_abs_grad, param.shape))
                total_params += flat_abs_grad.numel() 
            
        k = int(self.drop_ratio * total_params)
        
        # 2. Concatenate all gradients to find global threshold
        all_flat_grads = torch.cat([item[1] for item in grad_items])
        
        if k <= 0:
             threshold = -1.0
        elif k >= len(all_flat_grads):
             threshold = float('inf')
        else:
            sorted_vals, _ = torch.sort(all_flat_grads)
            threshold = sorted_vals[k - 1]
            
        # 3. Generate mask for each parameter
        for name, flat_abs_grad, shape in grad_items:
            layer_mask = (flat_abs_grad <= threshold).float()
            mask[name] = layer_mask.view(shape)
            
        return mask

    def __call__(self, clients: List[torch.Tensor], args=None) -> nn.Module: 
        """Run Federated Learning with FedMemPrune algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        backdoor_acc_threshold = getattr(self.args, 'unlearn_acc_threshold', 1 / self.args.num_classes)
        first_in_threshold = True
        print(">>> Backdoor accuracy threshold for unlearning: {:.4f}".format(backdoor_acc_threshold))
        best_acc = 0.0
        best_unlearn_acc = 1.0
        best_epoch = 0
        best_model_state = None

        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            best_acc = self.evaluate(args.test_loader)
            best_backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {best_acc:.4f}, Backdoor Test Accuracy: {best_backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_backdoor_acc:.4f}")


        start_time = time.time()
        best_time, best_comm = 0.0, 0.0     # record the time and communication cost for the best model

        # Unlearning Phase1: Locate memorization parameters
        memorization_mask = self.get_memorization_mask(
            DataLoader(
                Subset(clients[0].global_dataset, [index for client in clients for index in client.remaining_indices]), 
                batch_size=getattr(self.args, 'batch_size', 64),
                shuffle=False,
            )
        )

        # Unlearning Phase2: Reset memorization parameters with Kaiming initialization
        for name, param in self.global_model.named_parameters():
            if name in memorization_mask:
                mask = memorization_mask[name].view(param.shape).cuda()
                reset_value = torch.empty_like(param.data)
                if reset_value.dim() >= 2:
                    torch.nn.init.kaiming_normal_(reset_value)
                else:
                    torch.nn.init.normal_(reset_value, mean=0.0, std=0.02)
                param.data = param.data * (1 - mask) + mask * reset_value

        # Unlearning Phase3: Fine-tuning on the remaining clients
        global_epochs = getattr(self.args, 'global_epochs', 80)
        
        for epoch in range(global_epochs):
            for client in clients: 
                if client.join_refine:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning(epoch=epoch) 

            unlearn_clients = [
                client for client in clients if client.join_refine
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)

                if backdoor_acc <= backdoor_acc_threshold: 
                    if acc >= best_acc or first_in_threshold: 
                        first_in_threshold = False
                        best_acc = acc
                        best_unlearn_acc = backdoor_acc
                        best_epoch = epoch + 1
                        best_model_state = self.global_model.state_dict()
                        best_time = time.time() - start_time
                        best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                        if args.save: 
                            save_model(best_model_state, args.save)
                elif backdoor_acc <= best_unlearn_acc: 
                    best_acc = acc
                    best_unlearn_acc = backdoor_acc
                    best_epoch = epoch + 1
                    best_model_state = self.global_model.state_dict()
                    best_time = time.time() - start_time
                    best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                    if args.save: 
                        save_model(best_model_state, args.save) 

                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}") 


        if best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)
        else:
            best_time = time.time() - start_time
            best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
        
        print(f"\n>>> Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")
        print(f"\n>>> Best time cost: {best_time:.4f} s")
        print(f">>> Best communication cost: {best_comm:.4f} MB\n")

        return self.global_model