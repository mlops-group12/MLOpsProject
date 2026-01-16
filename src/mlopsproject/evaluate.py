"""
Model Evaluation Module

This module provides functionality for evaluating trained models on test data.
It automatically selects the latest versioned model from local storage or GCS.
It computes test metrics using PyTorch Lightning and logs results.
"""

import sys
import os
import glob
import tempfile
import torch
from mlopsproject.model import CNN
from mlopsproject.data import get_dataloaders
from mlopsproject.visualize import plot_confusion_matrix
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import wandb
import matplotlib.pyplot as plt
from google.cloud import storage


@hydra.main(
    config_path="../../configs",
    config_name="base_config",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Evaluate a trained CNN model on test data.

    Steps:
    1. Loads train/validation/test splits from local folder or GCS
    2. Initializes CNN model
    3. Automatically selects the latest versioned model
    4. Loads model weights
    5. Evaluates on test data and logs metrics
    6. Optionally logs to WandB

    Args:
        cfg (DictConfig): Hydra configuration object
    """
    # -------------------------
    # Load test data
    # -------------------------
    _, _, test_data = get_dataloaders(
        gcs_bucket=cfg.gcs.bucket,
        gcs_folder=cfg.gcs.data_folder
    )

    model = CNN()

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
    # Select latest model
    # -------------------------
    model_version = os.getenv("MODEL_TIMESTAMP")
    if model_version:
        local_model_path = f"models/model_{model_version}.pt"
    else:
        # fallback: pick latest if timestamp not provided
        local_models = glob.glob("models/model_*.pt")
        local_model_path = max(local_models, key=os.path.getctime) if local_models else None


    gcs_model_path = None
    if cfg.gcs.bucket and cfg.gcs.model_folder:
        client = storage.Client(project="active-premise-484209-h0")
        bucket = client.bucket(cfg.gcs.bucket)
        blobs = list(bucket.list_blobs(prefix=cfg.gcs.model_folder))
        model_blobs = [b for b in blobs if b.name.endswith(".pt")]
        if model_blobs:
            # Pick latest model by name (timestamped)
            latest_blob = max(model_blobs, key=lambda b: b.name)
            tmp_dir = tempfile.mkdtemp()
            gcs_model_path = os.path.join(tmp_dir, latest_blob.name.split("/")[-1])
            print(f"Downloading latest model from GCS: {latest_blob.name} -> {gcs_model_path}")
            latest_blob.download_to_filename(gcs_model_path)

    # Decide which model to load: local preferred, fallback to GCS
    if local_model_path:
        print(f"Loading latest local model: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path))
        model_version = os.path.basename(local_model_path).split("_")[-1].replace(".pt", "")
    elif gcs_model_path:
        model.load_state_dict(torch.load(gcs_model_path))
        model_version = os.path.basename(gcs_model_path).split("_")[-1].replace(".pt", "")
    else:
        print("No model found locally or in GCS!")
        sys.exit(0)

    # -------------------------
    # Log model version to WandB
    # -------------------------
    if cfg.wandb.enabled:
        wandb.run.name = f"faces_{model_version}"

    # -------------------------
    # Evaluate
    # -------------------------
    trainer = Trainer(logger=logger)
    _ = trainer.test(model, dataloaders=test_data)

    # -------------------------
    # Confusion matrix
    # -------------------------
    preds = torch.cat(model.test_preds)
    targets = torch.cat(model.test_targets)

    class_names = ["angry", "fear", "happy", "sad", "surprise"]

    conf_fig = plot_confusion_matrix(preds, targets, class_names, normalize=False)
    if cfg.wandb.enabled:
        wandb.log({"confusion_matrix": wandb.Image(conf_fig)})
    plt.close(conf_fig)

    # Optional normalized confusion matrix
    plot_confusion_matrix(preds, targets, class_names, normalize=True)


if __name__ == "__main__":
    main()
