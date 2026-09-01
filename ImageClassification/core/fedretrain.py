"""
:file: fedretrain.py
:date: 2025-07-16
:description: Federated Retraining for Unlearning
"""


from typing import Optional, List
import time

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import GradScaler, autocast

from .federatedbase import Client, Server
from models import get_model_class
from utils import Communicator, save_model


class ClientFedRetrain(Client): 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Gradient Balancing(FedGB) Algorithm

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

        self.join_unlearning = self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices)

        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.join_refine else None

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Local unlearning method for FedGB algorithm

        :param loader: Dataloader for unlearning, defaults to unlearning_loader
        """
        if self.join_refine: 
            loader = self.remaining_loader
            
            self.local_model.train()

            scaler = GradScaler() if self.args.dataset in ['tiny_imagenet', 'imagenet100'] else None

            for epoch in range(getattr(self.args, 'local_epochs', 1)):
                for data, target in loader:
                    data, target = data.cuda(), target.cuda()
                    self.opt.zero_grad()
                    if scaler:
                        with autocast():
                            output = self.local_model(data)
                            loss = self.criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(self.opt)
                        scaler.update()
                    else:
                        output = self.local_model(data)
                        loss = self.criterion(output, target)
                        loss.backward()
                        self.opt.step()

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedGB algorithm

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        return self.local_unlearning(loader)


class ServerFedRetrain(Server): 
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedGB Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)
        # reinitialize the global model as a random one if pretrained model is not a warmup model

        # if self.args.pretrain_model is not end with "warmup"
        if self.args.pretrain_model is None or not self.args.pretrain_model.endswith("warmup"):
            model_class = get_model_class(args.model)
            self.global_model = model_class()
            self.global_model.cuda() 


    def __call__(self, clients: List[ClientFedRetrain], args=None) -> nn.Module:
        """Run Federated Learning with FedGB algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """

        best_acc = 0.0
        best_unlearn_acc = 1.0
        best_epoch = 0
        best_model_state = None
        best_time = 0.0
        best_comm = 0.0

        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            best_acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {best_acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epochs: {best_epoch}, Best Accuracy: {best_acc:.4f}")

        start_time = time.time()
        for epoch in range(getattr(self.args, 'global_epochs', 100)):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'unlearn_local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'): 
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                if acc >= best_acc:
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
        