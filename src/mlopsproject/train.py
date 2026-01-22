"""
Model Training Module

Trains a CNN using PyTorch Lightning on a pre-pulled dataset.
Always saves and uploads a single canonical model artifact: model-latest.pt
"""

from mlopsproject.model import CNN
from mlopsproject.data import get_dataloaders

from pytorch_lightning import Trainer
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

import torch
import os
import datetime

from google.cloud import storage
import wandb

@hydra.main(
    config_path="../../configs",
    config_name="base_config",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Orchestrates model training:
    1. Loads dataset (assumed pre-pulled)
    2. Initializes model
    3. Trains using PyTorch Lightning
    4. Saves model as model-latest.pt
    5. Uploads model to GCS (overwrites)
    """

    # -------------------------
    # Training parameters
    # -------------------------
    max_epochs = cfg.epochs
    lr = cfg.lr

    print("Starting training job")
    print(f"Epochs: {max_epochs}, LR: {lr}")

    # -------------------------
    # Data loading
    # -------------------------
    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(num_workers=2)

    # -------------------------
    # Model initialization
    # -------------------------
    model = CNN(learning_rate=lr)
    print("Model initialized")
    print("Device:", model.device)

    # -------------------------
    # WandB logger
    # -------------------------
    logger = None
    run_tag = os.getenv(
        "MODEL_TIMESTAMP",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    if cfg.wandb.enabled:
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
        )
        wandb.run.name = f"faces_{run_tag}"

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # -------------------------
    # Save model (latest-only)
    # -------------------------
    local_model_path = "models/model-latest.pt"
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    if cfg.save_model:
        torch.save(model.state_dict(), local_model_path)
        print(f"Model saved locally at {local_model_path}")

    # -------------------------
    # Upload to GCS (overwrite)
    # -------------------------
    if cfg.gcs.bucket and cfg.gcs.model_folder:
        try:
            client = storage.Client(project=cfg.gcs.project)
            bucket = client.bucket(cfg.gcs.bucket)

            blob = bucket.blob(
                f"{cfg.gcs.model_folder}/model-latest.pt"
            )
            blob.upload_from_filename(local_model_path)

            print(
                f"Model uploaded to "
                f"gs://{cfg.gcs.bucket}/{cfg.gcs.model_folder}/model-latest.pt"
            )

        except Exception as e:
            print("Failed to upload model to GCS")
            print("Reason:", e)

    print("Training job completed successfully")


if __name__ == "__main__":
    main()
