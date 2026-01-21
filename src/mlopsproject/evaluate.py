"""
Model Evaluation Module

This module evaluates a trained CNN model on test data.
It automatically pulls the dataset via DVC, selects the latest versioned model
from local storage or GCS, computes test metrics, and logs results to WandB if enabled.
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
from google.cloud import storage
import matplotlib.pyplot as plt


@hydra.main(
    config_path="../../configs",
    config_name="base_config",
    version_base=None,
)
def main(cfg: DictConfig):
    """Evaluate a trained CNN model on test data."""

    # -------------------------
    # Load test data (DVC-first)
    # -------------------------
    _, _, test_data = get_dataloaders(num_workers=2)

    # -------------------------
    # Initialize model
    # -------------------------
    model = CNN()

    # -------------------------
    # Setup WandB logger
    # -------------------------
    logger = None
    if cfg.wandb.enabled:
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
        )

    # -------------------------
    # Determine model version / path
    # -------------------------
    model_version = os.getenv("MODEL_TIMESTAMP")
    local_model_path = None
    if model_version:
        local_model_path = f"models/model_{model_version}.pt"
    else:
        # fallback: pick latest local model
        local_models = glob.glob("models/model_*.pt")
        if local_models:
            local_model_path = max(local_models, key=os.path.getctime)

    # -------------------------
    # Optionally download latest model from GCS
    # -------------------------
    gcs_model_path = None
    if cfg.gcs.bucket and cfg.gcs.model_folder:
        try:
            client = storage.Client(project="active-premise-484209-h0")
            bucket = client.bucket(cfg.gcs.bucket)
            blobs = [b for b in bucket.list_blobs(prefix=cfg.gcs.model_folder) if b.name.endswith(".pt")]
            if blobs:
                latest_blob = max(blobs, key=lambda b: b.name)
                tmp_dir = tempfile.mkdtemp()
                gcs_model_path = os.path.join(tmp_dir, os.path.basename(latest_blob.name))
                print(f"Downloading latest model from GCS: {latest_blob.name} -> {gcs_model_path}")
                latest_blob.download_to_filename(gcs_model_path)
        except Exception as e:
            print("Skipping GCS model download. Reason:", e)

    # -------------------------
    # Load model weights
    # -------------------------
    if local_model_path and os.path.exists(local_model_path):
        print(f"Loading latest local model: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path))
        model_version = os.path.basename(local_model_path).split("_")[-1].replace(".pt", "")
    elif gcs_model_path:
        print(f"Loading model from GCS: {gcs_model_path}")
        model.load_state_dict(torch.load(gcs_model_path))
        model_version = os.path.basename(gcs_model_path).split("_")[-1].replace(".pt", "")
    else:
        print("No model found locally or in GCS!")
        sys.exit(1)

    # -------------------------
    # Log model version to WandB
    # -------------------------
    if cfg.wandb.enabled and logger is not None:
        logger.experiment.name = f"faces_{model_version}"

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

    # Raw confusion matrix
    conf_fig = plot_confusion_matrix(preds, targets, class_names, normalize=False)
    if cfg.wandb.enabled and logger is not None:
        import wandb

        wandb.log({"confusion_matrix": wandb.Image(conf_fig)})
    plt.close(conf_fig)

    # Normalized confusion matrix
    plot_confusion_matrix(preds, targets, class_names, normalize=True)


if __name__ == "__main__":
    main()
