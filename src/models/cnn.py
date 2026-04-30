# src/models/cnn.py

import torch.nn as nn
import torch.nn.functional as F

# Compact CNN for FEMNIST (28 x 28 grayscale, 62 classes)
class FEMNIST_CNN(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.conv2 = nn.Conv2d(4, 16, 5, padding=2)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # -> 14 × 14 × 4
        x = self.pool(F.relu(self.conv2(x))) # -> 7 x 7 x 16
        x = x.view(-1, 16 * 7 * 7)           # -> 784
        x = F.relu(self.fc1(x))              # -> 64
        x = self.fc2(x)                      # -> 62
        return x


# Compact CNN for CIFAR-10 (32 x 32 RGB, 10 classes)
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # -> 16 x 16 x 32
        x = self.pool(F.relu(self.conv2(x))) # -> 8 x 8 x 64
        x = F.relu(self.conv3(x))            # -> 8 x 8 x 64
        x = x.mean(dim=(2, 3))               # global avg pool -> 64
        return self.fc(x)                    # -> 10