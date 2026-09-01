# Federated Unlearning with Contractive Unlearning Perturbation

[![Venue: ICASSP 2027](https://img.shields.io/badge/Venue-ICASSP%202027-1f77b4)]()
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)]()

This is the official code repository for our paper *Federated Unlearning with Contractive Unlearning Perturbation* (FedCUP), which is currently under review at the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2027).

## 🖥️ Experimental Platform

All experiments were conducted on a server equipped with **4 × NVIDIA GeForce RTX 3080 Ti GPUs** (CUDA 11.6, Ubuntu Linux).

## 🔧 Environment Setup

The code is developed and tested with **Python 3.9**. We recommend using a conda environment:

```bash
# 1) create and activate the environment
conda create -n fedcup python=3.9 -y
conda activate fedcup

# 2) install the CUDA-specific PyTorch wheels (CUDA 11.6, matching our setup)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 3) install the remaining dependencies
pip install -r requirements.txt
```

The full dependency list is available in [`requirements.txt`](requirements.txt). If you use a different CUDA version, please install the matching `torch` / `torchvision` wheels from [pytorch.org](https://pytorch.org/get-started/previous-versions/) before step 3.

## 🧑‍💻 Code

The repository contains two self-contained sub-projects:

```
├── ImageClassification/        # deep image classification (CNN / ResNet)
│   ├── unlearning.py           # main federated unlearning entry
│   ├── init.py                 # argument & YAML configuration parsing
│   ├── config/                 # per-dataset configurations
│   ├── core/                   # FedCUP (ours) + FedAvg & all baselines
│   ├── datasets/               # CIFAR-10/100, SVHN, Tiny-ImageNet loaders
│   ├── models/                 # CNN, MLP, ResNet architectures
│   ├── scripts/                # reproducible run scripts
│   └── utils/                  # communicators & compressors (EF14/EF21/EFFACE, TopK, ...)
└── LogisticRegression/         # binary logistic regression (MNIST 3 vs. 8)
    ├── pretrain.py             # federated pretraining (FedAvg)
    ├── unlearn.py              # federated unlearning (fedgb / fedgd / fedcup)
    ├── retrain.py              # retraining-from-scratch baseline
    ├── optimizer.py            # losses, gradients, and update rules
    ├── scripts/                # pretrain / unlearn / retrain / delta ablation
    └── config/config.yml       # default configuration
```

### Dataset 

Default dataset loading path is `./data/` (can be changed in `ROOT_FOLDER` in [`ImageClassification/datasets/__init__.py`](ImageClassification/datasets/__init__.py) and `MNIST_ROOT` in [`LogisticRegression/loader.py`](LogisticRegression/loader.py))

### Logistic Regression

[`LogisticRegression/`](LogisticRegression) provides a convex / strongly-convex testbed on **binary MNIST** (digits 3 vs. 8) for studying the behavior of FedCUP, including an ablation over the contraction coefficient `delta`.

```bash
cd LogisticRegression

bash scripts/pretrain.sh         # federated pretraining (FedAvg)
bash scripts/unlearn.sh          # federated unlearning (fedgb / fedgd / fedcup)
bash scripts/retrain.sh          # retraining-from-scratch baseline
bash scripts/ablation_deltas.sh  # ablation over the contraction coefficient
```

### Image Classification

[`ImageClassification/`](ImageClassification) evaluates federated unlearning on **CIFAR-10**, **CIFAR-100**, **SVHN** and **Tiny-ImageNet** with backdoor-based unlearning (pixel-pattern triggers). It implements the proposed **FedCUP** as well as a comprehensive set of baselines.

```bash
cd ImageClassification

# 1) Federated pretraining (FedAvg) — produces the pretrained global model
bash scripts/cifar10/pretrain.sh

# 2) Federated unlearning with FedCUP (layer-wise, sweeps the contraction coefficient delta)
bash scripts/fedcup.sh          # or: bash scripts/delta.sh

# 3) ablation over the contraction coefficient delta
bash scripts/delta.sh
```

