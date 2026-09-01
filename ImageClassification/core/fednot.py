"""
:file: fednot.py
:date: 2025-08-21 
:description: Federated Unlearning via Weight Negation 
:src: [NoT: Federated Unlearning via Weight Negation (CVPR 2025)](https://arxiv.org/pdf/2503.05657), Section 10, 11. 
"""


from typing import Optional, List 
import time

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from utils import Communicator, save_model

class ClientFedNot(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Unlearning via Weight Negation (FedNot) Algorithm

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

    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None:
        super().set_dataloader(batch_size, shuffle)
        self.unlearning_indices = unlearning_indices
        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = True 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices)

        self.unlearning_loader = DataLoader(
            Subset(self.global_dataset, self.unlearning_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.is_unlearner else None
        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.join_refine else None

    def local_unlearning(self, loader = None):
        return self.local_refinement(loader)
    
    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
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


class ServerFedNot(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedNot Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args=args)
        negate_layers_dict = {
            'mnist': 1,
            'fashion_mnist': 1,
            'cifar10': 1,
            'svhn': 1,
            'cifar100': 1,
            'tiny_imagenet': 1
        }
        self.negate_layers = negate_layers_dict.get(self.args.dataset, 1)

    def __call__(self, clients: List[ClientFedNot], args=None) -> nn.Module:
        """Run Federated Learning with FedGD algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        # Unlearning Phase: Weight Negation
        with torch.no_grad():
            count = 0
            for name, params in self.global_model.named_parameters(): 
                # only negate the first few convolutional layers, skip some layers like batchnorm
                if 'conv' in name.lower() and 'weight' in name.lower():
                    if count < self.negate_layers:
                        params.data = -params.data
                        count += 1
                    else:
                        break  

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

        unlearn_epochs = getattr(self.args, 'global_epochs', 100)
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_refine:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_refine
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 1 == 0: 
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