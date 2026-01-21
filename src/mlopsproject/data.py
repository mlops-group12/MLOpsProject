"""
Data Loading Module (DVC + local support)
"""

import os
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import subprocess


def pull_dvc_data():
    """
    Pull the DVC-tracked dataset to the path recorded in the repo (data/).
    Returns the dataset folder path.
    """
    print("Pulling dataset via DVC...")
    try:
        # Pull all DVC-tracked files (including data.dvc)
        subprocess.run(["dvc", "pull"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ DVC pull failed. Please check your DVC setup and remotes.")

    dataset_path = "data"  # path recorded in data.dvc
    if not os.path.isdir(dataset_path):
        raise RuntimeError(
            f"No valid dataset folder found at {dataset_path} after DVC pull."
        )

    print(f"✅ Dataset ready at {dataset_path}")
    return dataset_path


def get_dataloaders(seed=0, num_workers=4, train_batch_size=64):
    """
    Create train, validation, and test DataLoaders.

    Uses DVC dataset if available; falls back to local folder if necessary.
    """
    np.random.seed(seed)

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # Pull DVC dataset first
    dataset_path = pull_dvc_data()

    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)

    # Split into train/val/test
    train_length = int(0.8 * len(dataset))
    val_length = int(0.9 * len(dataset))
    indices = np.random.permutation(len(dataset))

    train_dataset = Subset(dataset, indices[:train_length])
    validation_dataset = Subset(dataset, indices[train_length:val_length])
    test_dataset = Subset(dataset, indices[val_length:])

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, num_workers=num_workers, batch_size=train_batch_size, persistent_workers=True
    )
    val_loader = DataLoader(
        validation_dataset, num_workers=num_workers, batch_size=train_batch_size, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, num_workers=num_workers, batch_size=train_batch_size, persistent_workers=True
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(validation_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
