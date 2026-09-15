""" 
:file: fedurge.py
:date: 2026-08-10
:description: Federated Unlearning with Contractive Unlearning Perturbation
"""
 
from copy import deepcopy
import random
import time
from typing import Optional, List 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np 
from torch.cuda.amp import autocast, GradScaler

from .federatedbase import Client, Server
from utils import Communicator, save_checkpoint, save_model


class ClientFedURGE(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Unlearning with Residual Gradient Estimation Algorithm
        
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

        # FedURGE local state
        # g_ri: remaining gradient estimator, initialized lazily on the first update
        self.grad_r_estimator = None
        # v_i: momentum estimator of the remaining gradient (FedURGE), initialized lazily
        self.momentum_estimator = None
        # delta: contraction parameter controlling the clip radius R = (delta/lambda_i) * ||d||
        self.delta = getattr(self.args, 'delta', 0.999)
        # momentum: momentum coefficient, v = momentum * v_prev + (1 - momentum) * grad_r  (favors history)
        self.momentum = getattr(self.args, 'momentum', 0.9)
        # layerwise: constrict the unlearning gradient per layer (default True).
        # When True, only layers whose gradient difference exceeds their own radius
        # get clipped; otherwise a single global radius is applied to the whole vector.
        self.layerwise = getattr(self.args, 'layerwise', True) 


    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None:
        super().set_dataloader(batch_size, shuffle)

        self.batch_size = batch_size
        self.unlearning_indices = unlearning_indices 
        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = True 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices) 
        self.gamma = len(self.unlearning_indices) / len(self.data_indices) # strike the balance between unlearning and refinement loss
        
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

        self.unlearning_iter = iter(self.unlearning_loader) if self.is_unlearner else None
        self.remaining_iter = iter(self.remaining_loader) if self.join_refine else None

    def _get_next_batch(self): 
        try:
            unlearn_data, unlearn_target = next(self.unlearning_iter)
        except StopIteration:
            self.unlearning_iter = iter(self.unlearning_loader)
            unlearn_data, unlearn_target = next(self.unlearning_iter)
        
        try:
            remain_data, remain_target = next(self.remaining_iter)
        except StopIteration:
            self.remaining_iter = iter(self.remaining_loader)
            remain_data, remain_target = next(self.remaining_iter)
        
        return (unlearn_data.cuda(non_blocking=True), unlearn_target.cuda(non_blocking=True),
                remain_data.cuda(non_blocking=True), remain_target.cuda(non_blocking=True))

    @torch.no_grad()
    def _update_local_model(self, grads: List[torch.Tensor]) -> None:
        """Directly update the local model in-place:  x <- x - lr * g.

        :param grads: list of per-parameter gradient tensors (same order as model.parameters())
        """
        lr = getattr(self.args, 'lr', 0.01)
        for p, g in zip(self.local_model.parameters(), grads):
            p.data.add_(g, alpha=-lr)

    @staticmethod
    def _global_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Compute the global (frobenius) norm of a list of per-parameter tensors.

        :param tensors: list of per-parameter gradient tensors
        :return: scalar norm
        """
        return torch.sqrt(sum(t.float().pow(2).sum() for t in tensors))

    @staticmethod
    def _clip_global_norm(tensors: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
        """Clip a list of per-parameter tensors (seen as one flattened vector) to a max norm.

        :param tensors: list of per-parameter gradient tensors
        :param max_norm: maximum norm of the flattened vector
        :return: clipped tensors (in-place scaling)
        """
        total_norm = ClientFedURGE._global_norm(tensors)
        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            for t in tensors:
                t.mul_(clip_coef)
        return tensors

    def local_unlearning(self, loader: Optional[DataLoader] = None, epoch: int = 0) -> None:
        """Local unlearning method for FedURGE (Federated Unlearning with Residual Gradient Estimation).

        Follows the README "Algorithm Overview" pseudocode, in both its global and
        layer-wise (default) variants (the model aggregation is still done by the
        server via FedAvg):
          initialize: v_i^0 = g_ri^0 = grad_f(x^0) = lambda*grad_u + (1-lambda)*grad_r
          - momentum estimator:    v^{t+1} = momentum * v^t + (1 - momentum) * grad_r
          - d            = v^{t+1} - g_ri^t
          - R            = (delta / lambda_i) * ||d||
          - g_ui^t       = d + clip(grad_u - d, R)
          - g_ri^{t+1}   = g_ri^t + (1 - lambda_i)(v^{t+1} - g_ri^t) + lambda_i * g_ui^t
          - local model  : x^{t+1} = x^t - lr * g_ri^{t+1}
        When `layerwise=True` (default), the constriction of g_ui is applied per
        layer with a per-layer radius; a layer whose gradient difference already
        satisfies ||grad_u - d|| <= (delta/lambda_i)||d|| is not clipped, skipping
        the clipping compute. When `layerwise=False`, a single global radius is used.
        g_ri and v are maintained as client-local state and are NOT communicated.

        :param loader: Dataloader for unlearning, defaults to None
        :param epoch: Current epoch, defaults to 0
        """
        self.local_model.train()

        lambda_i = self.gamma  # unlearning ratio for this client 

        if self.is_unlearner:
            try: 
                unlearn_data, unlearn_target = next(self.unlearning_iter)
            except StopIteration:
                self.unlearning_iter = iter(self.unlearning_loader)
                unlearn_data, unlearn_target = next(self.unlearning_iter)
            unlearn_data, unlearn_target = unlearn_data.cuda(non_blocking=True), unlearn_target.cuda(non_blocking=True)

            try: 
                remain_data, remain_target = next(self.remaining_iter)
            except StopIteration:
                self.remaining_iter = iter(self.remaining_loader)
                remain_data, remain_target = next(self.remaining_iter)
            remain_data, remain_target = remain_data.cuda(non_blocking=True), remain_target.cuda(non_blocking=True)

            data = torch.cat([unlearn_data, remain_data], dim=0)

            output = self.local_model(data)
            output_u, output_r = output[:len(unlearn_data)], output[len(unlearn_data):]

            loss_u, loss_r = self.criterion(output_u, unlearn_target), self.criterion(output_r, remain_target)

            params = list(self.local_model.parameters())

            grad_u = torch.autograd.grad(-loss_u, params, retain_graph=True)
            grad_r = torch.autograd.grad(loss_r, params, retain_graph=False)
            grad_u = [g.detach() for g in grad_u]
            grad_r = [g.detach() for g in grad_r]

            if self.momentum_estimator is None or self.grad_r_estimator is None:
                self.momentum_estimator = [g.clone() for g in grad_r]
                # initialize grad_r_estimator to zero tensors
                self.grad_r_estimator = [torch.zeros_like(g) for g in grad_r]
            else:
                torch._foreach_mul_(self.momentum_estimator, self.momentum)
                torch._foreach_add_(self.momentum_estimator, grad_r, alpha=1 - self.momentum)

            v = self.momentum_estimator
            g_ri = self.grad_r_estimator

            d = [torch.sub(v_p, g_p) for v_p, g_p in zip(v, g_ri)] 

            if self.layerwise:
                diffs = [torch.sub(g_u, d_p) for g_u, d_p in zip(grad_u, d)]
                diff_norms = [torch.norm(x) for x in diffs]
                d_norms = [torch.norm(x) for x in d]
                coeff = self.delta / lambda_i
                g_ui = []
                for i, (diff, dn, d_n) in enumerate(zip(diffs, diff_norms, d_norms)):
                    R = coeff * d_n
                    if dn > R:
                        diff.mul_(R / (dn + 1e-8))
                        g_ui.append(d[i] + diff) 
                    else:
                        g_ui.append(grad_u[i]) # skip clipping for this layer
            else:
                diff = [g_u - d_p for g_u, d_p in zip(grad_u, d)]
                diff_norm = ClientFedURGE._global_norm(diff)
                R = ClientFedURGE._global_norm(d) * (self.delta / lambda_i)
                if diff_norm > R:
                    ClientFedURGE._clip_global_norm(diff, R.item())
                    g_ui = [d_p + diff_p for d_p, diff_p in zip(d, diff)]
                else:
                    g_ui = grad_u                # skip clipping over the whole vector

            # --- update remaining gradient estimator ---
            torch._foreach_mul_(g_ri, lambda_i)
            torch._foreach_add_(g_ri, v, alpha=1 - lambda_i)
            torch._foreach_add_(g_ri, g_ui, alpha=lambda_i)

            # --- local model update: x <- x - lr * g_ri^{t+1} ---
            self._update_local_model(g_ri)
        else:
            try:
                remaining_data, remaining_target = next(self.remaining_iter)
            except StopIteration:
                self.remaining_iter = iter(self.remaining_loader)
                remaining_data, remaining_target = next(self.remaining_iter)
            remaining_data, remaining_target = remaining_data.cuda(non_blocking=True), remaining_target.cuda(non_blocking=True)

            remaining_output = self.local_model(remaining_data)
            loss = self.criterion(remaining_output, remaining_target)

            grad_r = [g.detach() for g in torch.autograd.grad(loss, self.local_model.parameters())]

            # non-unlearner (lambda_i = 0): keep the momentum update v, and g_ri tracks v
            # (with lambda_i = 0 the FedURGE update g_ri^{t+1} = v^{t+1} collapses to v)
            if self.momentum_estimator is None or self.grad_r_estimator is None:
                self.momentum_estimator = [g.clone() for g in grad_r]
                self.grad_r_estimator = [g.clone() for g in grad_r]
            else:
                torch._foreach_mul_(self.momentum_estimator, self.momentum)
                torch._foreach_add_(self.momentum_estimator, grad_r, alpha=1 - self.momentum)

                # copy the momentum estimator to the remaining gradient estimator (g_ri tracks v)
                torch._foreach_copy_(self.grad_r_estimator, self.momentum_estimator)

            # --- local model update: x <- x - lr * g_ri ---
            self._update_local_model(self.grad_r_estimator) 


class ServerFedURGE(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for Federated Unlearning with Residual Gradient Estimation Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args=args)


    # @torch.no_grad()
    # def aggregate(self, clients: List[Client]) -> None: 
    #     """Aggregate client models, do not need communicate function to accelarate the aggregation.

    #     :param clients: list of Client
    #     """
    #     uploads = [client.local_model.state_dict() for client in clients]
    #     global_state_dict = self.global_model.state_dict()
    #     for key in global_state_dict.keys():
    #         global_state_dict[key] = torch.mean(
    #             torch.stack([upload[key].float() for upload in uploads]), dim=0
    #         )
    #     self.global_model.load_state_dict(global_state_dict)


    def __call__(self, clients: List[torch.Tensor], args=None) -> nn.Module: 
        """Run Federated Learning with FedRL algorithm

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
        start_comm = 0.0
        best_time, best_comm = 0.0, 0.0     # record the time and communication cost for the best model
        # Unlearning Phase1: Balanced Forgetting 
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 80)
        
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning(epoch=epoch) 

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

        if best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)
        else:
            best_time = time.time() - start_time
            best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
        
        print(f"\n>>> Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")
        print(f"\n>>> Best time cost: {best_time:.4f} s")
        print(f">>> Best communication cost: {best_comm:.4f} MB\n")

        return self.global_model