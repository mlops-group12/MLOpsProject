"""
Model Training Module

This module provides functionality for training CNN models using PyTorch Lightning.
It handles data loading, model initialization, training loop, and model checkpointing.
"""

from mlopsproject.model import CNN
from mlopsproject.data import get_dataloaders
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import hydra
import torch
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs",
    config_name="base_config",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Train a CNN model on the dataset.

    This function orchestrates the training process:
    1. Loads and prepares data (train/validation splits)
    2. Initializes the CNN model with specified hyperparameters
    3. Sets up optional WandB logging
    4. Trains the model using PyTorch Lightning
    5. Saves model weights if configured

    Args:
        cfg (DictConfig): Hydra configuration object.

    Note:
        - Model weights are saved to 'models/model_weights_latest.pt' if save_model is True
        - Training and validation metrics are logged via PyTorch Lightning
        - If WandB is enabled, metrics are also logged to WandB
    """
    max_epochs = cfg.epochs
    lr = cfg.lr
    train_data, validation_data, _ = get_dataloaders()

    model = CNN(learning_rate=lr)
    print("device:", model.device)

    # setup WandB logger only if enabled
    if cfg.wandb.enabled:
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
        )
    else:
        logger = None

    trainer = Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=validation_data)

    # from datetime import datetime

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.save_model:
        torch.save(model.state_dict(), "models/model_weights_latest.pt")


if __name__ == "__main__":
    main()
