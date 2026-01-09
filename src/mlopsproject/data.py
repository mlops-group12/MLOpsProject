from pathlib import Path

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

DATA_PATH = "data/"

def get_data_splits(seed=0, num_workers=0, train_batch_size=64) -> tuple[DataLoader, DataLoader, DataLoader]: 
    cifar10_path = Path(DATA_PATH) / "cifar-10"

    transform = transforms.Compose([transforms.ToTensor()])

    if not cifar10_path.exists():
        cifar10_path.mkdir(parents=True, exist_ok=True)

    # construct new indices
    np.random.seed(seed)
    indices = np.random.choice(50000, 50000)
    train_indices = indices[:40000]
    validation_indices = indices[40000:]


    main_data = CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)

    train_data = Subset(main_data, train_indices)
    validation_data = Subset(main_data, validation_indices)

    test_data = CIFAR10(
        root=cifar10_path,
        train=False,
        download=True,
        transform=transform
    )

    # ------------------------------------------------------------------
    # convert to dataloaders:
    # ------------------------------------------------------------------

    train_dataloader = DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=train_batch_size,
        persistent_workers=False,
    )

    test_dataloader = DataLoader(
        test_data,
        num_workers=num_workers,
        persistent_workers=False,
    )

    validation_dataloader = DataLoader(
        validation_data,
        num_workers=num_workers,
        persistent_workers=False,
    )

    return train_dataloader, test_dataloader, validation_dataloader

if __name__ == "__main__":
    train_data, test_data, validation_data = get_data_splits()
    pass
