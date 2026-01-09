from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

DATA_PATH = "data/"

def get_data_splits(seed=0) -> tuple[Dataset, Dataset]: 
    cifar10_path = Path(DATA_PATH) / "cifar-10"

    if not cifar10_path.exists():
        cifar10_path.mkdir()

    # construct new indices
    np.random.seed(seed)
    indices = np.random.choice(50000, 50000)
    train_indices = indices[:40000]
    validation_indices = indices[40000:]


    main_data = CIFAR10(root=cifar10_path, train=True, download=True, transform=None)

    train_data = Subset(main_data, train_indices)
    validation_data = Subset(main_data, validation_indices)
    
    test_data = CIFAR10(root=cifar10_path, train=False, download=True, transform=None)

    return train_data, validation_data, test_data

if __name__ == "__main__":
    train_data, test_data = get_data_splits()
    pass
