"""
:file: fedosd.py
:date: 2026-08-15
:src: [Federated Unlearning with Gradient Descent and Conflict Mitigation (AAAI 2025)](https://arxiv.org/pdf/2412.20200), [GitHub](https://github.com/zibinpan/FedOSD)
:description: FedOSD (Federated Unlearning with Orthogonal Steepest Descent) Algorithm Implementation

The original FedOSD targets the *client removal* unlearning task. Each client is
either an unlearning client or a retained client. During the *unlearning phase*, the
server aggregates the unlearning gradient `gu` (from unlearning clients, computed with
the Unlearning Cross-Entropy (UCE) loss), and its steepest descent direction that is
orthogonal to the subspace spanned by the retained clients' gradients `gr_locals` is
taken as the global update direction `d` (Orthogonal Steepest Descent Direction).
During the *recovery phase*, only retained clients keep training, and a *model
reverting* guard is applied which removes the component of the retained gradients that
is parallel to the moving direction `ga` (current model - initial model) so that the
model does not drift back to the original (e.g. backdoored) model.

This implementation migrates FedOSD to the *sample removal* task used in this codebase:
every client holds both unlearning samples and remaining samples. In the unlearning
phase each client produces an unlearning gradient `gu_i` (UCE loss on its unlearning
samples) and a retained gradient `gr_i` (cross-entropy loss on its remaining samples).
The server then:
    gu      = mean(gu_i)                                  # aggregate unlearning gradient
    d       = Proj_orth(gu | span{gr_i})                  # nearest orthogonal steepest descent
    d       = d / ||d|| * ||gu||                          # keep the norm of the unlearning
    x_{t+1} = x_t - lr * d
    
In the recovery phase, remaining samples only are trained with the model-reverting guard.
Different from the client-driven aggregation of most algorithms in this folder, FedOSD is
a server-driven algorithm: clients compute local gradients and the server directly applies
the aggregated update direction to the global model.
"""

from copy import deepcopy
import os
from typing import Optional, List, Tuple

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from utils import Communicator, save_model

# ---------------------------------------------------------------------------
# Unlearning Cross-Entropy (UCE) loss (used by the unlearning clients)
# ---------------------------------------------------------------------------
class UnLearningCELoss:
    """UCE loss proposed in FedOSD.

    The model's prediction probability for the (forgotten) target class is pushed
    towards 1/2, i.e. the model is made maximally uncertain about the unlearned samples.

    loss = -log(1 - softmax(pred) / 2) masked over the true class.
    """

    def __init__(self, ignore_index: int = -100, reduction: str = 'mean') -> None:
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        class_num = int(pred.shape[1])

        ignore_indices = torch.where(target == self.ignore_index)[0]
        if len(ignore_indices) > 0:
            target = target.clone()
            target[ignore_indices] = 0
            target_enc = F.one_hot(target, class_num)
            target_enc[ignore_indices, 0] = 0
        else:
            target_enc = F.one_hot(target, class_num)

        pred = F.softmax(pred, dim=-1)
        if self.reduction == 'none':
            loss = -torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1)
        elif self.reduction == 'mean':
            loss = -torch.mean(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
        elif self.reduction == 'sum':
            loss = -torch.sum(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        return loss


class ClientFedOSD(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for the FedOSD Algorithm

        :param global_dataset: Global dataset for the client
        :param data_indices: Indices of the local dataset for the client
        :param local_model: Local model
        :param client_id: Client index, defaults to 0
        :param comm: Communicator for communication between client and server
        :param unlearning_indices: Indices of the local samples to be forgotten
        :param args: Other arguments
        """
        super().__init__(global_dataset, data_indices, local_model, client_id, comm, args=args)

        self.set_dataloader(
            batch_size=getattr(self.args, 'batch_size', len(self.data_indices)),
            shuffle=getattr(self.args, 'shuffle', True),
            unlearning_indices=unlearning_indices
        )

        # UCE loss is used for the unlearning samples
        self.uce_criterion = UnLearningCELoss()

        # Per-client gradients computed during the unlearning phase (flattened vectors).
        # These are consumed by the server to build the orthogonal steepest descent
        # direction and are NOT communicated through the Communicator.
        self.grad_u: Optional[torch.Tensor] = None   # gradient w.r.t. unlearning samples (UCE)
        self.grad_r: Optional[torch.Tensor] = None   # gradient w.r.t. remaining samples (CE)

    def set_dataloader(self, batch_size: int = 64, shuffle: bool = True, unlearning_indices: List[int] = []) -> None:
        self.batch_size = batch_size
        self.unlearning_indices = unlearning_indices

        super().set_dataloader(batch_size, shuffle)

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

    def _grad_to_flat_vector(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of per-parameter gradients into a single flat vector.

        :param gradients: list of gradient tensors
        :return: flattened gradient vector
        """
        return torch.cat([g.detach().view(-1) for g in gradients])

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Compute the per-client unlearning and retained gradients (FedOSD, unlearning phase).

        The gradient of the UCE loss w.r.t. the unlearning samples is stored in
        ``self.grad_u`` and the gradient of the cross-entropy loss w.r.t. the remaining
        samples is stored in ``self.grad_r``. The server aggregates these to build the
        orthogonal steepest descent direction later, so no local optimizer step is
        performed here (server-driven update).

        Like the original FedOSD's batch-gradient-descent (bgd) step, the gradient is
        accumulated over the *whole* local (unlearning / remaining) dataset, so that each
        global round corresponds to a full-dataset gradient-descent step.

        :param loader: Dataloader for unlearning, defaults to None
        """
        params = list(self.local_model.parameters())

        if self.is_unlearner:
            self.grad_u = None
            for unlearn_data, unlearn_target in self.unlearning_loader:
                unlearn_data, unlearn_target = unlearn_data.cuda(), unlearn_target.cuda()
                output_u = self.local_model(unlearn_data)
                loss_u = self.uce_criterion(output_u, unlearn_target)
                grad_u = torch.autograd.grad(loss_u, params)
                flat = self._grad_to_flat_vector(grad_u)
                self.grad_u = flat if self.grad_u is None else self.grad_u + flat

        if self.join_refine:
            self.grad_r = None
            for remain_data, remain_target in self.remaining_loader:
                remain_data, remain_target = remain_data.cuda(), remain_target.cuda()
                output_r = self.local_model(remain_data)
                loss_r = self.criterion(output_r, remain_target)
                grad_r = torch.autograd.grad(loss_r, params)
                flat = self._grad_to_flat_vector(grad_r)
                self.grad_r = flat if self.grad_r is None else self.grad_r + flat
        else:
            self.grad_r = None

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Compute the retained gradient for the recovery phase (FedOSD, recovery phase).

        Stores the flattened gradient of the remaining samples in ``self.grad_r`` so the
        server can apply the model-reverting guard and drive the update (server-driven).

        The gradient is accumulated over the whole remaining dataset (bgd step), matching
        the original FedOSD recovery framework.

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        if loader is None:
            loader = self.remaining_loader

        self.grad_r = None
        for remain_data, remain_target in loader:
            remain_data, remain_target = remain_data.cuda(), remain_target.cuda()
            output = self.local_model(remain_data)
            loss = self.criterion(output, remain_target)
            grad = torch.autograd.grad(loss, self.local_model.parameters())
            flat = self._grad_to_flat_vector(grad)
            self.grad_r = flat if self.grad_r is None else self.grad_r + flat


class ServerFedOSD(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for the FedOSD Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)

        self.initial_g_norm = None
        self.init_model_params = None   # flattened model parameters at the start of unlearning

    # ------------------------------------------------------------------
    # FedOSD core math
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_model_params(model: nn.Module) -> torch.Tensor:
        """Flatten all model parameters into a single vector.

        :param model: the model
        :return: flattened parameter vector
        """
        return torch.cat([p.data.detach().view(-1).float() for p in model.parameters()])

    @staticmethod
    def cal_pseudoinverse(matrix: torch.Tensor) -> torch.Tensor:
        """Moore-Penrose pseudo-inverse of a matrix computed via SVD.

        :param matrix: input matrix
        :return: pseudo-inverse matrix
        """
        return torch.pinverse(matrix, rcond=1e-6)

    def get_nearest_oth_d(self, gr_locals: torch.Tensor, gu: torch.Tensor) -> torch.Tensor:
        """Compute the nearest orthogonal steepest descent direction.

        Given the retained clients' gradients stacked as rows of `gr_locals` (matrix A)
        and the aggregate unlearning gradient `gu` (vector c), the update direction is

            d = c - A^T (A A^T)^+ (A c)

        which is the projection of `gu` onto the orthogonal complement of the subspace
        spanned by the retained gradients, i.e. the steepest descent direction of the
        unlearning objective that does not conflict with the retained clients.

        :param gr_locals: matrix whose rows are the retained clients' gradients (A)
        :param gu: aggregate unlearning gradient (c)
        :return: orthogonal steepest descent direction d
        """
        A = gr_locals
        A_T = A.T
        c = gu

        AAT_1 = self.cal_pseudoinverse(A @ A_T)
        Ac = A @ c.reshape(-1, 1)
        AAT_1_Ac = AAT_1 @ Ac

        d = c - (A_T @ AAT_1_Ac).reshape(-1)
        return d

    # ------------------------------------------------------------------
    # Server-driven gradient update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _grad_update_model(self, d: torch.Tensor, lr: float) -> None:
        """Update the global model: x <- x - lr * d.

        :param d: flattened update direction
        :param lr: learning rate
        """
        offset = 0
        for p in self.global_model.parameters():
            numel = p.numel()
            p.data.add_(d[offset:offset + numel].view_as(p).float(), alpha=-lr)
            offset += numel

    def _collect_gradients(
        self,
        clients: List[ClientFedOSD],
        attr: str,
    ) -> Tuple[List[torch.Tensor], List[ClientFedOSD]]:
        """Collect a per-client flattened gradient attribute as tensors.

        :param clients: list of clients
        :param attr: the attribute holding the flattened gradient, e.g. 'grad_u'
        :return: list of gradient tensors and the corresponding active clients
        """
        grads, active = [], []
        for client in clients:
            grad = getattr(client, attr)
            if grad is None:
                continue
            grads.append(grad.detach().reshape(1, -1))
            active.append(client)
        return grads, active

    def __call__(self, clients: List[ClientFedOSD], args=None) -> nn.Module:
        """Run federated unlearning with the FedOSD algorithm.

        Two phases:
          1) Unlearning phase: build the orthogonal steepest descent direction `d` from
             the per-client unlearning/retained gradients and directly apply it to the
             global model: x <- x - lr * d.
          2) Recovery phase: train only on the remaining samples with the model-reverting
             guard to recover utility without drifting back to the original model.

        :param clients: List of clients participating in the federated unlearning
        :param args: Other arguments
        :return: Updated global model
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
        best_time, best_comm = 0.0, 0.0

        # remember the model at the start of unlearning (used by the model-reverting guard)
        self.init_model_params = self._flatten_model_params(self.global_model)

        # =================================================================
        # Phase 1: Unlearning with Orthogonal Steepest Descent
        # =================================================================
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 80)
        unlearn_lr = getattr(self.args, 'lr', 0.01)

        for epoch in range(unlearn_epochs):
            for client in clients:
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    client.local_unlearning()

            # aggregate the unlearning gradient and the retained gradients
            gu_vecs, _ = self._collect_gradients(clients, 'grad_u')
            gr_vecs, _ = self._collect_gradients(clients, 'grad_r')

            # gu = mean of the unlearning gradients
            gu = torch.mean(torch.cat(gu_vecs, dim=0), dim=0) if gu_vecs else \
                torch.zeros_like(self._flatten_model_params(self.global_model))
            gu_norm = torch.norm(gu)
            g_norm = gu_norm

            if getattr(self.args, 'orthogonal', os.environ.get('FEDOSD_ORTHOGONAL', '1') == '1'):
                # d = steepest descent direction of gu orthogonal to span{gr_locals}
                gr_locals = torch.cat(gr_vecs, dim=0) if gr_vecs else gu.reshape(1, -1)
                d = self.get_nearest_oth_d(gr_locals, gu)
            else:
                # diagnostic: use the plain unlearning gradient (skip orthogonal projection)
                d = gu

            # keep the norm of the unlearning gradient
            d_norm = torch.norm(d)
            if d_norm > 0:
                d = d / d_norm * g_norm

            # update the global model: x <- x - lr * d
            self._grad_update_model(d, unlearn_lr)

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

        # restore the best unlearned model before the recovery phase
        if best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)

        # =================================================================
        # Phase 2: Recovery with model-reverting guard
        # =================================================================
        refine_epochs = getattr(self.args, 'refine_epochs', 0)
        refine_lr = getattr(self.args, 'ft_lr', None) or getattr(self.args, 'lr', 0.01)

        if refine_epochs > 0:
            start_time += time.time() - (best_time + start_time)  # reset time for the recovery phase
            start_comm += sum([client.get_communication_cost(unit='MB') for client in clients]) - best_comm
            if best_model_state is not None:
                self.global_model.load_state_dict(best_model_state)

            ga = self._flatten_model_params(self.global_model) - self.init_model_params
            ga_norm = torch.norm(ga)

            for epoch in range(refine_epochs):
                for client in clients:
                    if client.join_refine:
                        client.set_model(self.global_model)
                        client.local_refinement()

                gr_vecs, _ = self._collect_gradients(clients, 'grad_r')

                # model-reverting guard: remove the component of each retained gradient
                # that is parallel to the unlearning direction ga
                grad_list = []
                for grad in gr_vecs:
                    g = grad.reshape(-1)
                    if ga_norm > 0:
                        g = g - (g @ ga / ga_norm ** 2) * ga
                    # normalize each retained gradient to a unit direction so that the
                    # per-round step is controlled by `refine_lr` alone (the raw CE
                    # gradient of a whole local dataset is orders of magnitude too large
                    # to be used directly with a plain lr).
                    g_norm = torch.norm(g)
                    if g_norm > 1e-8:
                        g = g / g_norm
                    grad_list.append(g.reshape(1, -1))

                d_refine = torch.mean(torch.cat(grad_list, dim=0), dim=0)

                # normalize the aggregated recovery direction to a unit vector
                d_norm = torch.norm(d_refine)
                if d_norm > 1e-8:
                    d_refine = d_refine / d_norm

                # update the global model with the recovery learning rate
                self._grad_update_model(d_refine, refine_lr)

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

