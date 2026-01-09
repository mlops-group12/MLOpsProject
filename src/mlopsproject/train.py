from .model import CNN
from .data import get_data_splits
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../../configs", config_name="base_config")
def main(cfg: DictConfig):
    max_epochs = cfg.epochs
    lr = cfg.lr
    train_data, validation_data, test_data = get_data_splits()

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

    trainer = Trainer(max_epochs=max_epochs,logger=logger)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=validation_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    main()