"""
:file: unlearn.py
:date: 2026-08-04 (create date) / 2026-08-04 (last modified date)
:description: Federated unlearning supporting fedgb, fedgd and fedcup algorithm. The server aggregates the gradients and updates the global model.
:src: [Federated Unlearning with Contractive Unlearning Perturbation]()
"""

from copy import deepcopy

import os

import numpy as np

from init import get_args
from utils import *
from optimizer import *
from loader import *


if __name__ == "__main__":
    args = get_args("config/config.yml")
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    is_save = args.save_ckpts
    if is_save:
        ckpt_dir = f"cache/ckpts/{args.unlearn_alg}_seed{seed}_alpha{args.alpha}"
        os.makedirs(ckpt_dir, exist_ok=True)

    num_clients = args.num_clients

    # load data
    train_loaders, test_loader, backdoor_test_loader, client_indices, client_unlearn_indices, train_dataset = load_dataloader(args)
    client_remain_indices = [list(set(indices) - set(unlearn_indices)) for indices, unlearn_indices in zip(client_indices, client_unlearn_indices)]

    Xs, ys = map(list, zip(*[dataloader_to_numpy(train_loader) for train_loader in train_loaders]))
    Xs_unlearn, ys_unlearn = map(list, zip(*[dataloader_to_numpy(DataLoader(Subset(train_dataset, indices=unlearn_indices), batch_size=args.batch_size, shuffle=False)) for unlearn_indices in client_unlearn_indices])) 

    Xs_remain, ys_remain = map(list, zip(*[dataloader_to_numpy(DataLoader(Subset(train_dataset, indices=remain_indices), batch_size=args.batch_size, shuffle=False)) for remain_indices in client_remain_indices]))

    # concatenate per-client data into the full remaining / unlearning training sets
    X_remain_all = np.concatenate(Xs_remain, axis=0)
    y_remain_all = np.concatenate(ys_remain, axis=0)
    X_unlearn_all = np.concatenate(Xs_unlearn, axis=0)
    y_unlearn_all = np.concatenate(ys_unlearn, axis=0)

    X_test, y_test = dataloader_to_numpy(test_loader)
    X_backdoor_test, y_backdoor_test = dataloader_to_numpy(backdoor_test_loader)

    # lambda_i = |D_i^u| / |D_i|: fraction of unlearning data on each client
    lam_us = [Xs_unlearn[i].shape[0] / Xs[i].shape[0] for i in range(num_clients)]
    print(f"Lambda_i (fraction of unlearning data) for each client: {lam_us}")

    lr = args.lr 
    lam = args.weight_decay
    epochs = args.global_epochs

    # load the pretrained global model, matching cache/models naming conventions
    #   - iid (partition == 'iid' or alpha >= 1e8): w_seed{seed}_iid.npy
    #   - non-iid: w_seed{seed}_alpha{alpha}.npy
    is_iid = args.partition == 'iid' or args.alpha >= 1e8
    if is_iid: 
        pretrain_name = f"cache/models/w_seed{args.seed}_iid.npy"
    else: 
        pretrain_name = f"cache/models/w_seed{args.seed}_alpha{args.alpha}.npy"
    w = np.load(pretrain_name) 


    gr_track = None
    error = None
    if args.unlearn_alg == 'fedcup':
        gr_track = []
        for i in range(num_clients):
            gr_track.append(np.zeros(w.shape[0]))

    for epoch in range(epochs): 
        if is_save:
            np.save(f"cache/ckpts/{args.unlearn_alg}_seed{seed}_alpha{args.alpha}/{epoch}.npy", w)

        deltas = []  
        diffs = []  # store the difference between the update and grad_r
        Rt = []
        Ut = []
        for i in range(num_clients): 
            ws_i = deepcopy(w) 
            grad_u = -grad_logistic(ws_i, Xs_unlearn[i], ys_unlearn[i], lam=lam)
            grad_r = grad_logistic(ws_i, Xs_remain[i], ys_remain[i], lam=lam)
            if args.unlearn_alg == 'fedgd': 
                delta = grad_r 
            elif args.unlearn_alg == 'fedgb':  
                delta = lam_us[i] * grad_u + (1 - lam_us[i]) * grad_r
            elif args.unlearn_alg == 'fedcup': 
                d = grad_r - gr_track[i]
                grad_u_used = grad_u  # the grad_u actually used by this variant 
                lam_i = lam_us[i]
                grad_u_adj = control_gd_u(d, grad_u, lam=lam_i, delta=args.delta)
                Ut.append(np.linalg.norm(grad_u_adj - grad_u))
                grad_u_used = grad_u_adj  # use the projected grad_u for the contraction check
                grad_f = lam_i * grad_u_adj + (1 - lam_i) * grad_r
                gr_track[i] = lam_i * gr_track[i] + grad_f
                delta = gr_track[i]
                norm_d = np.linalg.norm(d) + 1e-12 
                Rt.append(norm_d)

            deltas.append(delta)
            diffs.append(np.linalg.norm(delta - grad_r))  # store the difference between the update and grad_r

        if (epoch + 1) % 1 == 0: 
            test_acc = evaluate_accuracy(w, X_test, y_test)
            backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test) 
            remain_loss = logistic_loss(w, X_remain_all, y_remain_all, lam=lam)
            unlearn_loss = -logistic_loss(w, X_unlearn_all, y_unlearn_all, lam=lam)
            diffs_mean = np.mean(diffs)
            if args.unlearn_alg == 'fedcup':
                Rt_mean = np.mean(Rt)
                Ut_mean = np.mean(Ut)
                print(f"Epoch {epoch}/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}, Diff: {diffs_mean:.4f}, Remain Loss: {remain_loss:.4f}, Unlearn Loss: {unlearn_loss:.4f}, R_t: {Rt_mean}, U_t: {Ut_mean}")
            else:
                print(f"Epoch {epoch}/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}, Diff: {diffs_mean:.4f}, Remain Loss: {remain_loss:.4f}, Unlearn Loss: {unlearn_loss:.4f}")

        # aggregate
        grad = np.mean(deltas, axis=0) 
        w = gradient_descent(w, grad, lr=lr)

    if is_save:
        np.save(f"cache/ckpts/{args.unlearn_alg}_seed{seed}_alpha{args.alpha}/{epochs}.npy", w)
        np.save(f"cache/models/w_{args.unlearn_alg}_seed{seed}_alpha{args.alpha}.npy", w)