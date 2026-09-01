"""
:file: pretrain.py
:date: 2026-08-04 (create date) / 2026-08-04 (last modified date)
:description: Pretrain a global logistic regression model on the binary MNIST 
    task over all clients and save the model weight to cache/models.
:src: [Federated Unlearning with Contractive Unlearning Perturbation]()
"""

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
    train_loaders, test_loader, backdoor_test_loader, client_indices, client_unlearn_indices, _ = load_dataloader(args)
    Xs, ys = map(list, zip(*[dataloader_to_numpy(train_loader) for train_loader in train_loaders]))
    X_test, y_test = dataloader_to_numpy(test_loader)
    X_backdoor_test, y_backdoor_test = dataloader_to_numpy(backdoor_test_loader)

    lr = args.lr 
    lam = args.weight_decay
    epochs = args.global_epochs

    w = np.zeros(Xs[0].shape[1])
    ws = [np.zeros(Xs[0].shape[1]) for _ in range(num_clients)]

    for epoch in range(epochs): 
        grads = []
        for i in range(num_clients): 
            ws[i] = w.copy()
            loss = logistic_loss(ws[i], Xs[i], ys[i], lam=lam)
            grads.append(grad_logistic(ws[i], Xs[i], ys[i], lam=lam))
        # aggregate
        grad = np.mean(grads, axis=0)
        w = gradient_descent(w, grad, lr=lr)

        if (epoch + 1) % 50 == 0: 
            test_acc = evaluate_accuracy(w, X_test, y_test)
            backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test)
            print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}")

    # save the pretrained model, matching cache/models naming conventions
    #   - iid (partition == 'iid' or alpha >= 1e8): w_seed{seed}_iid.npy
    #   - non-iid: w_seed{seed}_alpha{alpha}.npy
    save_folder = "cache/models"
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)
    is_iid = args.partition == 'iid' or args.alpha >= 1e8
    if is_iid: 
        save_name = f"w_seed{args.seed}_iid.npy"
    else: 
        save_name = f"w_seed{args.seed}_alpha{args.alpha}.npy"
    np.save(os.path.join(save_folder, save_name), w)
    print(f">>> Saved pretrained model to {os.path.join(save_folder, save_name)}")
