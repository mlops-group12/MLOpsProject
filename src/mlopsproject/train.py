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
