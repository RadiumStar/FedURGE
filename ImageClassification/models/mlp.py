"""
:file: mlp.py
:date: 2025-07-14
:description: Multi-Layer Perceptron Model for Federated Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_channel=1, input_dim=784, num_class=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_channel * input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1) if self.fc3.out_features == 1 else x
    

def MLP_MNIST(input_channel=1, input_dim=784, num_class=10):
    """MLP model for MNIST dataset"""
    return MLP(input_channel=input_channel, input_dim=input_dim, num_class=num_class)

