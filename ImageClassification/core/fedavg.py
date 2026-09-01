"""
:file: fedavg.py
:date: 2025-07-11
:description: Federated Averaging Algorithm Implementation(Pretrained Model for Unlearning)
"""

from typing import List
import time
 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 
from torch.cuda.amp import autocast, GradScaler

from .federatedbase import Client, Server
from utils import Communicator, save_model

class ClientFedAvg(Client): 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None): 
        """Federated Learning Client for FedAvg Algorithm

        :param global_dataset: Global dataset for the client
        :param data_indices: Indices of the local dataset for the client
        :param local_model: Local model
        :param client_id: Client index, defaults to 0
        :param unlearning_indices: useless
        :param args: Other arguments
        """
        super().__init__(global_dataset, data_indices, local_model, client_id, comm, args=args)

        # if args.dataset in ['tiny_imagenet', 'imagenet100']: 
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.opt, 
        #         T_max=getattr(self.args, 'global_epochs', 500),
        #         eta_min=1e-6
        #     )     # using for warmup
        #     # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     #     self.opt, 
        #     #     milestones=[30, 50], 
        #     #     gamma=0.1
        #     # )
        # elif args.dataset in ['cifar10']:
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.opt, 
        #         T_max=getattr(self.args, 'global_epochs', 500),
        #         eta_min=1e-6
        #     )

        self.set_dataloader(
            batch_size=getattr(self.args, 'batch_size', len(self.local_dataset)), 
            shuffle=getattr(self.args, 'shuffle', True), 
            unlearning_indices=unlearning_indices
        )

    def set_dataloader(self, batch_size: int = 64, shuffle: bool = True, unlearning_indices: List[int] = []) -> None:
        """Set dataloader for local dataset

        :param batch_size: batch size for the dataloader, defaults to 64
        :param shuffle: whether to shuffle the dataset, defaults to True
        :param unlearning_indices: indices of samples to be unlearned
        """
        self.train_loader = DataLoader(
            self.local_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=8, 
            pin_memory=True, 
            drop_last=True
        )


    def train(self, loader: DataLoader = None, epoch: int = 0) -> None:
        """Train local model using FedAvg algorithm

        :param loader: Dataloader for training, defaults to train_loader
        :param epoch: Current epoch number
        """
        if loader is None:
            loader = self.train_loader
        
        self.local_model.train()

        scaler = GradScaler() if self.args.dataset in ['tiny_imagenet', 'imagenet100'] else None # use mixed precision training for large datasets

        for _ in range(getattr(self.args, 'local_epochs', 1)):
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                self.opt.zero_grad()

                if scaler: 
                    with autocast():
                        output_main = self.local_model(data)
                        loss = self.criterion(output_main, target)
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
    

class ServerFedAvg(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedAvg Algorithm

        :param global_model: Global model for the server
        :param dataset: Evaluation dataset
        :param args: Other arguments
        """
        super().__init__(global_model, args)

    def __call__(self, clients: list[ClientFedAvg], args=None) -> nn.Module:
        """Run Federated Learning with FedAvg algorithm

        :param clients: List of client instances
        :param args: Additional arguments for training, defaults to None
        :return: Updated global model
        """
        best_acc = 0.0
        best_epoch = 0
        best_model_state = None

        global_epochs = getattr(self.args, 'global_epochs', 200)

        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            best_acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {best_acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epochs: {best_epoch}, Best Accuracy: {best_acc:.4f}")

        start_time = time.time()
        best_time, best_comm = 0.0, 0.0     # record the time and communication cost for the best model
        for epoch in range(global_epochs):
            for client in clients:
                client.set_model(self.global_model)
                client.train()
            
            self.aggregate(clients)

            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                if acc >= best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    best_model_state = self.global_model.state_dict()
                    best_time = time.time() - start_time
                    best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                    if args.save: 
                        save_model(best_model_state, args.save)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epochs: {best_epoch}, Best Accuracy: {best_acc:.4f}")

        # if best_model_state is not None, turn best_model_state into best_model and return it
        if best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)
        else:
            best_time = time.time() - start_time
            best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
        
        print(f"\n>>> Best time cost: {best_time:.4f} s")
        print(f">>> Best communication cost: {best_comm:.4f} MB\n")
            
        return self.global_model 
    