from torch.utils.data import DataLoader, TensorDataset
from mlopsproject.model import CNN
from pytorch_lightning import Trainer
import torch


def dummy_dataloader():
    x = torch.randn(8, 1, 64, 64)
    y = torch.randint(0, 5, (8,))
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
    )


def test_lightning_training_loop():
    model = CNN()
    trainer = Trainer(
        fast_dev_run=True,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(
        model,
        train_dataloaders=dummy_dataloader(),
        val_dataloaders=dummy_dataloader(),
    )
