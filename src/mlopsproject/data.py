"""
Data Loading Module

This module provides utilities for loading and preprocessing image data,
creating train/validation/test splits, and generating PyTorch DataLoaders.
"""

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import os


def get_dataloaders(seed=0, num_workers=9, train_batch_size=64):
    """
    Create train, validation, and test DataLoaders from image folder dataset.

    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        num_workers (int, optional): Number of worker processes for data loading.
            Defaults to 9.
        train_batch_size (int, optional): Batch size for training DataLoader.
            Defaults to 64.

    Returns:
        tuple: A tuple containing three DataLoaders:
            - train_dataloader: DataLoader for training data
            - validation_dataloader: DataLoader for validation data
            - test_dataloader: DataLoader for test data

    """
    np.random.seed(seed)

    data_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Sizing
            transforms.Grayscale(),  # Making Black and White
            transforms.ToTensor(),  # Convert torch tensor
        ],
    )

    gcs_path = "/gcs/data-face-emotions/data"
    local_path = os.path.join(os.getcwd(), "data")

    path = gcs_path if os.path.isdir(gcs_path) else local_path

    print("Using dataset path:", path)
    dataset = datasets.ImageFolder(root=path, transform=data_transform)


    # split into indices
    train_length = int(0.8 * len(dataset))

    val_length = int(0.9 * len(dataset))

    indices = np.random.choice(len(dataset), len(dataset), replace=False)

    train_dataset = Subset(dataset, indices[:train_length])
    validation_dataset = Subset(dataset, indices[train_length:val_length])
    test_dataset = Subset(dataset, indices[val_length:])
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(validation_dataset))
    print("Number of test samples:", len(test_dataset))

    # convert to dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=train_batch_size,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        persistent_workers=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return train_dataloader, validation_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders()
