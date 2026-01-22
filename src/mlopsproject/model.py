"""
Model Definition Module

This module contains the CNN model architecture for image classification.
The model is implemented as a PyTorch Lightning module for easy training and evaluation.
"""

import torch.nn as nn
from torch import Tensor
from torch import optim
from pytorch_lightning import LightningModule
import torch


class CNN(LightningModule):
    """
    Convolutional Neural Network for grayscale emotion classification.

    Architecture:
        - Feature extractor: 3 convolutional blocks with batch normalization
          and max pooling, reducing spatial dimensions from 64x64 to 8x8
        - Classifier: Fully connected layers with dropout for regularization

    Input shape:  (N, 1, 64, 64) - Batch of grayscale images
    Output shape: (N, 5) - Logits for 5 emotion classes
    """

    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
    ) -> None:
        """
        Initialize the CNN model.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 5.
            learning_rate (float, optional): Learning rate for the optimizer.
                Defaults to 1e-4.
        """
        super().__init__()
        self.save_hyperparameters()

        # Feature extractor: 64x64 -> 32x32 -> 16x16 -> 8x8
        filter_coef = 4
        self.features = nn.Sequential(
            nn.Conv2d(1, filter_coef, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_coef),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_coef, filter_coef, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_coef),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(filter_coef, filter_coef * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_coef * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_coef * 2, filter_coef * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_coef * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(filter_coef * 2, filter_coef * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_coef * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        # infer flatten dim safely
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (N, 1, 64, 64)

        Returns:
            Tensor: Output logits of shape (N, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for PyTorch Lightning.

        Args:
            batch (tuple[Tensor, Tensor]): Tuple of (data, target) tensors
            batch_idx (int): Index of the current batch

        Returns:
            Tensor: Training loss value
        """
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        Validation step for PyTorch Lightning.

        Args:
            batch (tuple[Tensor, Tensor]): Tuple of (data, target) tensors
        """
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        Test step for PyTorch Lightning.

        Args:
            batch (tuple[Tensor, Tensor]): Tuple of (data, target) tensors
        """
        data, target = batch
        preds = self(data)

        self.test_preds.append(preds.argmax(dim=-1).cpu())
        self.test_targets.append(target.cpu())

        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for PyTorch Lightning.

        Returns:
            optim.Adam: Adam optimizer with the configured learning rate
        """
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def on_test_start(self):
        self.test_preds = []
        self.test_targets = []
