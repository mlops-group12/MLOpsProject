from pathlib import Path

import typer
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

DATA_PATH = "data/"

def get_data_splits() -> tuple[Dataset, Dataset]: 
    cifar10_path = Path(DATA_PATH) / 'cifar-10'

    if not cifar10_path.exists():
        cifar10_path.mkdir()

    train_data = CIFAR10(root=cifar10_path, train=True, download=True, transform=None)
    test_data = CIFAR10(root=cifar10_path, train=False, download=True, transform=None)

    return train_data, test_data

if __name__ == "__main__":
    typer.run(get_data_splits)
