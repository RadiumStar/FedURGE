"""
:file: unlearning.py
:date: 2025-07-15
:description: federated unlearning main script
"""

import time

from torch.utils.data import DataLoader 

from init import get_args
from core import get_client_class, get_server_class, Server, Client
from datasets import *
from models import get_model_class
from utils import *


if __name__ == "__main__": 
    # parse arguments and configurations
    args = get_args()
    seed = set_seed(args.seed)
    device = set_cuda(args.cuda)
    args.device = device 

    dataset_name = args.dataset
    num_client = args.num_clients
    unlearn_select = args.unlearn_select
    target_label = args.target_label
    src_label = args.src_label
    unlearn_perc = args.unlearn_perc
    
    unlearn_client_perc = args.unlearn_client_perc
    num_unlearn_clients = int(num_client * unlearn_client_perc)
    all_client_indices = list(range(num_client))
    
    unlearn_clients = set(random.sample(all_client_indices, num_unlearn_clients))
    unlearn_percs = [unlearn_perc if i in unlearn_clients else 0.0 for i in range(num_client)]

    # load dataset and model
    train_dataset = get_dataset(dataset_name, train=True)
    test_dataset = get_dataset(dataset_name, train=False)
    backdoor_test_dataset = get_dataset(dataset_name, train=False)

    model_class = get_model_class(args.model)
    global_model = model_class()
    if args.pretrain_model is not None: 
        global_model = load_model(global_model, args.pretrain_model, device=device, folder="cache/models")
        print(f">>> Pretrained model loaded from {args.pretrain_model}.pth")

    client_indices, client_unlearn_indices = load_client_data_indices(args)

    if client_indices is None or client_unlearn_indices is None:
        print(f">>> Client data indices not found, generating new client data indices")
        client_indices = get_client_indices(train_dataset, num_client, partition=args.partition, alpha=args.alpha, num_class=args.num_classes)

        client_unlearn_indices = [get_unlearn_indices(train_dataset, client_indice, unlearn_select, unlearn_perc, target_label, src_label) for client_indice, unlearn_perc in zip(client_indices, unlearn_percs)]

        save_client_data_indices(client_indices, client_unlearn_indices, args)

    if unlearn_select == 'backdoor': 
        train_dataset = get_backdoor_dataset(train_dataset, [
            unlearn_index for client_unlearn_indice in client_unlearn_indices for unlearn_index in client_unlearn_indice
        ], target_label=target_label, src_label=src_label , trigger_size=args.trigger_size, trigger_type=args.trigger_type, distance=args.trigger_distance)

        backdoor_test_dataset = get_backdoor_dataset(backdoor_test_dataset, backdoor_ratio=1.0, target_label=target_label, src_label=src_label, only_backdoor=True, trigger_size=args.trigger_size, trigger_type=args.trigger_type, distance=args.trigger_distance)
        # visualize_imgs(backdoor_test_dataset, num_imgs=5, save_path=f"img/{args.dataset}_backdoor_samples.pdf")
        # exit(0)  # [note] exit after visualization for backdoor test samples

    print(f'>>> Dataset: {dataset_name} has been loaded to {num_client} clients')

    # set federated system 
    server_class = get_server_class(args.unlearn_method)
    client_class = get_client_class(args.unlearn_method)
    server: Server = server_class(global_model, args)

    clients: list[Client] = [
        client_class(
            train_dataset, 
            client_indice, 
            model_class(), 
            client_id=i, 
            comm=get_communicator_class(
                communicator_name=args.communicator, 
                compressor_name=args.compressor, 
                eta=args.eta,
                k=args.k, 
                sparsity=args.sparsity, 
                bits=args.bits,
                is_init_compressed=args.is_init_compressed, 
            ),
            unlearning_indices=client_unlearn_indice, 
            args=args
        ) for i, (client_indice, client_unlearn_indice) in enumerate(zip(client_indices, client_unlearn_indices))
    ]
    print(f'>>> federated system with {len(clients)} clients has been set up') 

    args.check = False  # [note]  `check` is only used for debugging, set to False for normal training, unless you want to store the intermediate models for all epochs, which is not recommended due to large storage cost
    
    args.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    args.backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=args.batch_size, shuffle=False)

    # start federated (un)learning   
    print(f">>> Start {args.unlearn_method} (un)learning")

    start_time = time.time()
    server(clients, args)
    end_time = time.time()
    print(f">>> Total time cost: {end_time - start_time:.4f} s")
    communication_cost = sum([client.get_communication_cost(unit='MB') for client in clients])
    print(f">>> Total communication cost: {communication_cost:.4f} MB\n")

    if args.save is not None: 
        save_model(server.global_model.state_dict(), args.save)
        print(f">>> Model saved to cache/models/{args.save}.pth\n")