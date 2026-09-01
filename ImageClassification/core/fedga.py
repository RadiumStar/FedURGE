"""
:file: fedga.py
:date: 2025-08-06
:description: Federated Gradient Ascent Algorithm for Unlearning
:src: [Federated Unlearning: Guarantee the Right of Clients to Forget](https://ieeexplore.ieee.org/document/9964015)
"""

from copy import deepcopy
from typing import Optional, List
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from utils import Communicator, save_model



class ClientFedGA(Client): 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Gradient Ascent(FedGA) Algorithm

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

        self.join_unlearning = bool(self.unlearning_indices) 
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

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Local unlearning method for FedGA algorithm

        :param loader: Dataloader for unlearning, defaults to unlearning_loader
        """
        self.local_model.train()
        self.opt.zero_grad()

        unlearn_data, unlearn_target = next(iter(self.unlearning_loader))
        unlearn_data, unlearn_target = unlearn_data.cuda(), unlearn_target.cuda()
        
        output = self.local_model(unlearn_data)
        loss = -self.criterion(output, unlearn_target)

        loss.backward()

        if self.args.clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.args.clipping)

        self.opt.step()

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedGA algorithm

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        if loader is None:
            loader = self.remaining_loader
        
        self.local_model.train()
        
        data, target = next(iter(loader))
        data, target = data.cuda(), target.cuda()
        self.opt.zero_grad()
        output = self.local_model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.opt.step()


class ServerFedGA(Server): 
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedGA Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)

    def __call__(self, clients: List[ClientFedGA], args=None) -> nn.Module:
        """Run Federated Learning with FedGA algorithm

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
        start_comm = 0
        best_time, best_comm = 0.0, 0.0     # record the time and communication cost for the best model
        # Unlearning Phase1: Gradient Ascent Unlearning
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 80)
        
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
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
                        best_model_state = deepcopy(self.global_model.state_dict())
                        best_time = time.time() - start_time
                        best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                        if args.save: 
                            save_model(best_model_state, args.save)
                elif backdoor_acc <= best_unlearn_acc: 
                    best_acc = acc
                    best_unlearn_acc = backdoor_acc
                    best_epoch = epoch + 1
                    best_model_state = deepcopy(self.global_model.state_dict())
                    best_time = time.time() - start_time
                    best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                    if args.save: 
                        save_model(best_model_state, args.save) 

                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")

        args.lr = args.ft_lr if args.ft_lr else args.lr

        if getattr(self.args, 'refine_epochs', 20) > 0:
            start_time += time.time() - (best_time + start_time)  # reset the start time for refinement phase
            start_comm += sum([client.get_communication_cost(unit='MB') for client in clients]) - best_comm  # reset the start communication cost for refinement phase
            if best_model_state is not None:
                self.global_model.load_state_dict(best_model_state)

        for client in clients: 
            if client.join_refine:
                client.set_optimizer(args)
                client.comm.reset()

        # Unlearning Phase2: Utility Refinement
        for epoch in range(getattr(self.args, 'refine_epochs', 20)):
            for client in clients:
                if client.join_refine:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_refinement()

            refine_clients = [
                client for client in clients if client.join_refine
            ]
            self.aggregate(refine_clients)


            if (epoch + 1) % 1 == 0: 
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)

                if backdoor_acc <= backdoor_acc_threshold: 
                    if acc >= best_acc or first_in_threshold: 
                        first_in_threshold = False
                        best_acc = acc
                        best_unlearn_acc = backdoor_acc
                        best_epoch = epoch + 1 + unlearn_epochs
                        best_model_state = deepcopy(self.global_model.state_dict())
                        best_time = time.time() - start_time
                        best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
                        if args.save: 
                            save_model(best_model_state, args.save)
                elif backdoor_acc <= best_unlearn_acc: 
                    best_acc = acc
                    best_unlearn_acc = backdoor_acc
                    best_epoch = epoch + 1 + unlearn_epochs
                    best_model_state = deepcopy(self.global_model.state_dict())
                    best_time = time.time() - start_time
                    best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
                    if args.save: 
                        save_model(best_model_state, args.save) 

                print(f"Global Epoch {epoch + unlearn_epochs + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")

        if best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)
        else:
            best_time = time.time() - start_time
            best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
        
        print(f"\n>>> Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")
        print(f"\n>>> Best time cost: {best_time:.4f} s")
        print(f">>> Best communication cost: {best_comm:.4f} MB\n")

        return self.global_model
        