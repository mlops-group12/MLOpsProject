"""
Model Evaluation Module

This module provides functionality for evaluating trained models on test data.
It loads a saved model checkpoint and computes test metrics using PyTorch Lightning.
"""

import sys

from mlopsproject.model import CNN
from mlopsproject.data import get_dataloaders
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import torch


@hydra.main(
    config_path="../../configs",
    config_name="base_config",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Evaluate a trained CNN model on test data.

    This function loads a saved model checkpoint from 'models/model_weights_latest.pt',
    evaluates it on the test dataset, and logs metrics. Optionally logs to Weights & Biases
    if enabled in the configuration.

    Args:
        cfg (DictConfig): Hydra configuration object

    Raises:
        SystemExit: Exits with code 0 if model weights file is not found.

    """
    _, _, test_data = get_dataloaders()

    model = CNN()

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
    try:
        model.load_state_dict(torch.load("models/model_weights_latest.pt"))
    except FileNotFoundError:
        print("Model weights not found!")
        sys.exit(0)

    trainer = Trainer(logger=logger)

    # run testing
    _ = trainer.test(model, dataloaders=test_data)
    # should return a dictionary of logged items


if __name__ == "__main__":
    main()
