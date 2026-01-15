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

    # -------------------------
    # Load data
    # -------------------------
    # Added GCS integration: use bucket/folder if available
    train_data, validation_data, _ = get_dataloaders(
        local_path="data",                     # fallback local path
        gcs_bucket=cfg.gcs.bucket,
        gcs_folder=cfg.gcs.data_folder
    )

    # -------------------------
    # Initialize model
    # -------------------------
    model = CNN(learning_rate=lr)
    print("device:", model.device)

    # -------------------------
    # Setup WandB logger
    # -------------------------
    if cfg.wandb.enabled:
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
        )
    else:
        logger = None

    # -------------------------
    # Train
    # -------------------------
    trainer = Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=validation_data)

    # -------------------------
    # Save model locally and to GCS
    # -------------------------
    if cfg.save_model:
        # Save locally
        local_model_path = "models/model_weights_latest.pt"
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        torch.save(model.state_dict(), local_model_path)
        print(f"Model saved locally at {local_model_path}")

        # Save to GCS (only add if cfg.gcs.bucket is defined)
        if cfg.gcs.bucket and cfg.gcs.model_folder:
            client = storage.Client()
            bucket = client.bucket(cfg.gcs.bucket)

            # Use MODEL_VERSION env var if set, else default to 'latest'
            model_version = os.getenv("MODEL_VERSION", "latest")
            blob_path = f"{cfg.gcs.model_folder}/model_{model_version}.pt"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_model_path)

            print(f"Model uploaded to GCS at gs://{cfg.gcs.bucket}/{blob_path}")


if __name__ == "__main__":
    main()
