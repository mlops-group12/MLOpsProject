"""
Data Loading Module (Pre-pulled dataset, no DVC at runtime)
"""

import os
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets


def get_dataset_path() -> str:
    """
    Returns the dataset path.

    Assumes data has been pre-pulled and exists locally.
    """
    dataset_path = os.getenv("DATA_DIR", "data")

    if not os.path.isdir(dataset_path):
        raise RuntimeError(
            f"Dataset directory not found at '{dataset_path}'. "
            "Make sure the dataset is present in the Docker image."
        )

    print(f"Using dataset at {dataset_path}")
    return dataset_path


def get_dataloaders(seed=0, num_workers=4, train_batch_size=64):
    """
    Create train, validation, and test DataLoaders.
    """
    np.random.seed(seed)

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset_path = get_dataset_path()

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=data_transform,
    )

    # Train / val / test split
    train_length = int(0.8 * len(dataset))
    val_length = int(0.9 * len(dataset))
    indices = np.random.permutation(len(dataset))

    train_dataset = Subset(dataset, indices[:train_length])
    validation_dataset = Subset(dataset, indices[train_length:val_length])
    test_dataset = Subset(dataset, indices[val_length:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
