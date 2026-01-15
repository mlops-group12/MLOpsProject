import torch.nn as nn
from torch import Tensor
from torch import optim
from pytorch_lightning import LightningModule
import torch


class CNN(LightningModule):
    """
    Grayscale emotion classifier
    Input:  (N, 1, 64, 64)
    Output: (N, 5)
    """

    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
    ) -> None:
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
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch: tuple[Tensor, Tensor]) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
