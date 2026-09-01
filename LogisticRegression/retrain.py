"""
:file: retrain.py
:date: 2026-08-17 (create date) / 2026-08-17 (last modified date)
:description: Retrain a global logistic regression model on the binary MNIST
    task over the REMAINING (non-unlearned) data of all clients, following the
    same federated training procedure as pretrain.py, and save the model weight
    to cache/models.
:src: [paper name] Federated Unlearning Compensation
"""

import os

import numpy as np

from init import get_args
from optimizer import *
from loader import *


if __name__ == "__main__": 
    args = get_args("config/config.yml")
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_clients = args.num_clients

    # load data
    train_loaders, test_loader, backdoor_test_loader, client_indices, client_unlearn_indices, train_dataset = load_dataloader(args)

    # compute the remaining (non-unlearned) data indices for each client
    client_remain_indices = [list(set(indices) - set(unlearn_indices))
                             for indices, unlearn_indices in zip(client_indices, client_unlearn_indices)]

    # per-client remaining data as numpy arrays
    Xs_remain, ys_remain = map(list, zip(*[
        dataloader_to_numpy(DataLoader(Subset(train_dataset, indices=remain_indices),
                                       batch_size=args.batch_size, shuffle=False))
        for remain_indices in client_remain_indices]))

    # concatenate per-client remaining data into the full remaining training set
    X_remain_all = np.concatenate(Xs_remain, axis=0)
    y_remain_all = np.concatenate(ys_remain, axis=0)

    X_test, y_test = dataloader_to_numpy(test_loader)
    X_backdoor_test, y_backdoor_test = dataloader_to_numpy(backdoor_test_loader)

    lr = args.lr 
    lam = args.weight_decay
    epochs = args.global_epochs

    # train from scratch on the remaining data (same scheme as pretrain.py)
    w = np.zeros(Xs_remain[0].shape[1])
    ws = [np.zeros(Xs_remain[0].shape[1]) for _ in range(num_clients)]

    for epoch in range(epochs): 
        grads = []
        for i in range(num_clients): 
            ws[i] = w.copy()
            grad = grad_logistic(ws[i], Xs_remain[i], ys_remain[i], lam=lam)
            grads.append(grad)
        # aggregate
        grad = np.mean(grads, axis=0)
        w = gradient_descent(w, grad, lr=lr)

        if (epoch + 1) % 1 == 0: 
            test_acc = evaluate_accuracy(w, X_test, y_test)
            backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test)
            remain_loss = logistic_loss(w, X_remain_all, y_remain_all, lam=lam)
            print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.4f}, "
                  f"Backdoor Test Accuracy: {backdoor_test_acc:.4f}, "
                  f"Remain Loss: {remain_loss:.4f}")

    # save the retrained model, matching cache/models naming conventions
    #   - iid (partition == 'iid' or alpha >= 1e8): w_retrain_seed{seed}_iid.npy
    #   - non-iid: w_retrain_seed{seed}_alpha{alpha}.npy
    save_folder = "cache/models"
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)
    is_iid = args.partition == 'iid' or args.alpha >= 1e8
    if is_iid: 
        save_name = f"w_retrain_seed{args.seed}_iid.npy"
    else: 
        save_name = f"w_retrain_seed{args.seed}_alpha{args.alpha}.npy"
    np.save(os.path.join(save_folder, save_name), w)
    print(f">>> Saved retrained model to {os.path.join(save_folder, save_name)}")
