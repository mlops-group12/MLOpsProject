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
import os
from google.cloud import storage
import wandb
import datetime


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
    train_data, validation_data, _ = get_dataloaders(
        gcs_bucket=cfg.gcs.bucket,
        gcs_folder=cfg.gcs.data_folder
    )

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

    # -------------------------
    # Model versioning (auto)
    # -------------------------
    model_version = os.getenv("MODEL_VERSION")
    if model_version is None:
        # Auto-generate timestamped version if not set
        model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Using MODEL_VERSION = {model_version}")

    # Local model path with version
    local_model_path = f"models/model_{model_version}.pt"
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    # Save model locally
    if cfg.save_model:
        torch.save(model.state_dict(), local_model_path)
        print(f"Model saved locally at {local_model_path}")

    if cfg.gcs.bucket and cfg.gcs.model_folder:
        try:
            client = storage.Client()
            bucket = client.bucket(cfg.gcs.bucket)
            blob = bucket.blob(f"{cfg.gcs.model_folder}/model_{model_version}.pt")
            blob.upload_from_filename(local_model_path)
            print(f"Model uploaded to GCS at gs://{cfg.gcs.bucket}/{cfg.gcs.model_folder}/model_{model_version}.pt")
        except Exception as e:
            print("Skipping GCS upload. Could not connect to Google Cloud Storage.")
            print("Reason:", e)

    # -------------------------
    # Log model version to WandB
    # -------------------------
    if cfg.wandb.enabled:
        wandb.run.name = f"faces_{model_version}"


if __name__ == "__main__":
    main()
