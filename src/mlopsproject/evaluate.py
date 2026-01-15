import sys

from mlopsproject.model import CNN
from mlopsproject.data import get_dataloaders
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
import torch
from mlopsproject.visualize import plot_confusion_matrix


@hydra.main(
    config_path="../../configs",
    config_name="base_config",
    version_base=None,
)
def main(cfg: DictConfig):
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

    # ---- CONFUSION MATRIX ----
    preds = torch.cat(model.test_preds)
    targets = torch.cat(model.test_targets)

    class_names = ["angry", "fear", "happy", "sad", "surprise"]

    plot_confusion_matrix(
        preds,
        targets,
        class_names,
        normalize=False,
    )

    plot_confusion_matrix(
        preds,
        targets,
        class_names,
        normalize=True,
    )


if __name__ == "__main__":
    main()
