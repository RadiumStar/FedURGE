"""
:file: cnn.py
:date: 2025-07-29
:description: Convolutional Neural Network (CNN) models for MNIST and Fashion MNIST datasets
"""

import torch.nn as nn


class CNN2(nn.Module):
    def __init__(self, input_channel=1, num_class=10):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_class)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x
    

class CNN3(nn.Module):
    def __init__(self, input_channel=1, num_class=10):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 7 * 7, num_class)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(self.act(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
    

class CNN(nn.Module): 
    def __init__(self, input_channel=3, num_class=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_class)

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, input_channel=1, num_class=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

        self.act = nn.Tanh()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def CNN_MNIST(input_channel=1, num_class=10):
    """Create a CNN model for MNIST dataset."""
    return CNN(input_channel=input_channel, num_class=num_class)


def CNN_FASHION_MNIST(input_channel=1, num_class=10):
    """Create a CNN model for Fashion MNIST dataset."""
    return CNN(input_channel=input_channel, num_class=num_class)

def LeNet5_MNIST(input_channel=1, num_class=10):
    """Create a LeNet-5 model for MNIST dataset."""
    return LeNet5(input_channel=input_channel, num_class=num_class)