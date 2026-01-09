from mlopsproject.model import CNN
from mlopsproject.data import get_data_splits
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

def main():
    max_epochs = 5
    train_data, validation_data, test_data = get_data_splits()

    model = CNN()
    print("device:", model.device)
    logger = pl.loggers.WandbLogger(project="dtu_mlops")

    trainer = Trainer(max_epochs=max_epochs,logger=logger)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=validation_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    main()