import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Input:  (N, 3, 32, 32)
    Output: (N, 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # -> (N, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (N, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),                               # -> (N, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> (N, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),                               # -> (N, 128, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # -> (N, 128*8*8)
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

