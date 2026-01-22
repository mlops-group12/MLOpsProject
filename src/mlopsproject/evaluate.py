"""
Model Evaluation Module

Evaluates the latest trained CNN model on the test dataset.
Assumes:
- Dataset is pre-pulled and available locally
- Model artifact is named `model-latest.pt`
"""

import os
import sys
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
    """
    Evaluates a trained CNN model on test data.
    """

    print("Starting evaluation job")

    # -------------------------
    # Load test data
    # -------------------------
    print("Loading dataset...")
    _, _, test_loader = get_dataloaders(num_workers=2)

    # -------------------------
    # Initialize model
    # -------------------------
    model = CNN()

    # -------------------------
    # WandB logger
    # -------------------------
    logger = None
    if cfg.wandb.enabled:
        logger = pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
        )
        logger.experiment.name = "faces_evaluation"

    # -------------------------
    # Load model (local first, then GCS)
    # -------------------------
    local_model_path = "models/model-latest.pt"
    model_loaded = False

    if os.path.exists(local_model_path):
        print(f"Loading local model: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path))
        model_loaded = True

    elif cfg.gcs.bucket and cfg.gcs.model_folder:
        try:
            print("Local model not found, attempting GCS download...")
            client = storage.Client(project="active-premise-484209-h0")
            bucket = client.bucket(cfg.gcs.bucket)

            blob = bucket.blob(
                f"{cfg.gcs.model_folder}/model-latest.pt",
            )

            if not blob.exists():
                raise FileNotFoundError("model-latest.pt not found in GCS")

            tmp_dir = tempfile.mkdtemp()
            gcs_model_path = os.path.join(tmp_dir, "model-latest.pt")
            blob.download_to_filename(gcs_model_path)

            print(f"Downloaded model from GCS to {gcs_model_path}")
            model.load_state_dict(torch.load(gcs_model_path))
            model_loaded = True

        except Exception as e:
            print("Failed to load model from GCS")
            print("Reason:", e)

    if not model_loaded:
        print("No model available for evaluation")
        sys.exit(1)

    # -------------------------
    # Evaluate
    # -------------------------
    trainer = Trainer(logger=logger)
    trainer.test(model, dataloaders=test_loader)

    # -------------------------
    # Confusion matrix
    # -------------------------
    preds = torch.cat(model.test_preds)
    targets = torch.cat(model.test_targets)

    class_names = ["angry", "fear", "happy", "sad", "surprise"]

    # Raw confusion matrix
    fig_raw = plot_confusion_matrix(
        preds,
        targets,
        class_names,
        normalize=False,
    )

    if cfg.wandb.enabled and logger is not None:
        import wandb

        wandb.log({"confusion_matrix_raw": wandb.Image(fig_raw)})

    plt.close(fig_raw)

    # Normalized confusion matrix
    fig_norm = plot_confusion_matrix(
        preds,
        targets,
        class_names,
        normalize=True,
    )

    if cfg.wandb.enabled and logger is not None:
        import wandb

        wandb.log({"confusion_matrix_normalized": wandb.Image(fig_norm)})

    plt.close(fig_norm)

    print("Evaluation completed successfully")


if __name__ == "__main__":
    main()
