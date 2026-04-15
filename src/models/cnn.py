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