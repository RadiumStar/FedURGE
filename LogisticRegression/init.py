"""
:file: init.py
:date: 2026-08-04 (create date) / 2026-08-04 (last modified date)
:description: Load configuration (YAML) and build the argument parser.
:src: [paper name] Federated Unlearning Compensation
"""

from typing import Optional

import argparse
import yaml

def get_yml_config(config_file: str) -> dict: 
    """ Load configuration from a YAML file

    :param config_file: Path to the YAML config file
    :return: Dictionary containing the configuration
    """
    config = {} 
    if config_file is None: 
        return config
    
    if config_file.endswith('.yml') or config_file.endswith('.yaml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else: 
        raise ValueError("Configuration file must be a YAML file with .yml or .yaml extension")
    
    return config


def get_args(config_file: Optional[str] = None) -> argparse.Namespace: 
    """ get arguments and configurations from a YAML file

    :param config_file: Path to the YAML config file
    :param desc: Description for the argument parser
    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description='Configuration for the federated unlearning project', add_help=False)
    
    parser.add_argument('--config', type=str, default=config_file, help='Path to the YAML config file')
    args, _ = parser.parse_known_args()

    if config_file is None:
        config_file = args.config
    config = get_yml_config(config_file)

    parser = argparse.ArgumentParser(parents=[parser])

    # General settings
    parser.add_argument('--seed', type=int, default=config.get('seed', 0), help='Random seed for reproducibility')
    parser.add_argument('--cuda', type=int, default=config.get('cuda', 0), help='CUDA device index')
    parser.add_argument('--save', type=str, default=config.get('save', None), help='Path to save models and results, if None, do not save')
    parser.add_argument('--save_ckpts', type=bool, default=config.get('save_ckpts', True), help='Whether to save per-epoch global model checkpoints during unlearning (default: True)')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default=config.get('dataset', 'binary_mnist'), help='Dataset name')
    parser.add_argument('--input_channel', type=int, default=config.get('input_channel', 1), help='Number of input channels for the dataset') 
    parser.add_argument('--input_size', type=int, default=config.get('input_size', 28), help='Input size for the dataset(resolution)')
    parser.add_argument('--num_classes', type=int, default=config.get('num_classes', 10), help='Number of classes in the dataset')

    # Unlearning settings 
    parser.add_argument('--unlearn_alg', type=str, default=config.get('unlearn_alg', 'fedcup'), choices=['fedgb', 'fedgd', 'fedcup'], help='Unlearning algorithm: fedgb | fedgd | fedcup')
    parser.add_argument('--delta', type=float, default=config.get('delta', 0.9), help='Contraction constant in (0,1) for the dynamic lambda of the proposed algorithm')
    parser.add_argument('--unlearn_select', type=str, default=config.get('unlearn_select', 'backdoor'), choices=['backdoor', 'class', 'random'], help='Unlearning selection method')
    parser.add_argument('--target_label', type=int, default=config.get('target_label', 1), help='Target label for class and backdoor unlearning')
    parser.add_argument('--src_label', type=int, default=config.get('src_label', None), help='Source label for backdoor unlearning, if None, attack all labels')
    parser.add_argument('--unlearn_perc', type=float, default=config.get('unlearn_perc', 0.05), help='Percentage of data to unlearn')
    parser.add_argument('--unlearn_client_perc', type=float, default=config.get('unlearn_client_perc', 1.0), help='Percentage of clients to unlearn')

    # Model settings
    parser.add_argument('--model', type=str, default=config.get('model', 'LogisticRegression'), help='Model architecture')
    parser.add_argument('--pretrain_model', type=str, default=config.get('pretrain_model', None), help='Path to pretrained model, if None, do not use a pretrained model')

    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default=config.get('optimizer', 'SGD'), choices=['SGD', 'Adam'], help='Optimizer type')
    parser.add_argument('--criterion', type=str, default=config.get('criterion', 'BCEWithLogitsLoss'), help='Loss function')

    parser.add_argument('--lr', type=float, default=config.get('lr', 0.1), help='Learning rate for optimizer')
    parser.add_argument('--ft_lr', type=float, default=config.get('ft_lr', None), help='Learning rate for fine-tuning, if None, use the same as lr')
    parser.add_argument('--momentum', type=float, default=config.get('momentum', 0.0), help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay', 0.0), help='Weight decay for optimizer')
    parser.add_argument('--beta1', type=float, default=config.get('beta1', 0.9), help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=config.get('beta2', 0.999), help='Beta2 for Adam optimizer')
    parser.add_argument('--clipping', type=float, default=config.get('clipping', 0.0), help='Gradient clipping value, if 0, no clipping')

    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 128), help='Batch size for training')

    # Federated settings 
    parser.add_argument('--num_clients', type=int, default=config.get('num_clients', 10), help='Number of clients in federated learning')
    parser.add_argument('--partition', type=str, default=config.get('partition', 'iid'), choices=['iid', 'non_iid'], help='Data partitioning strategy')
    parser.add_argument('--alpha', type=float, default=config.get('alpha', 0.5), help='Dirichlet distribution parameter for non-iid partitioning')

    parser.add_argument('--global_epochs', type=int, default=config.get('global_epochs', 200), help='Number of global epochs for aggregation')
    parser.add_argument('--local_epochs', type=int, default=config.get('local_epochs', 1), help='Number of local epochs for each client')
    parser.add_argument('--unlearn_epochs', type=int, default=config.get('unlearn_epochs', 80), help='Number of epochs for unlearning')
    parser.add_argument('--refine_epochs', type=int, default=config.get('refine_epochs', 0), help='Number of epochs for model refinement after unlearning') 

    return parser.parse_args() 

