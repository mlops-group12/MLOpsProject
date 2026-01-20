"""
Data Loading Module (DVC + local support)
"""

import os
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import subprocess


def pull_dvc_data(dvc_file: str) -> str:
    """
    Pull a DVC-tracked dataset to its recorded path.
    Returns the folder path containing the dataset.
    """
    try:
        print(f"Pulling dataset via DVC ({dvc_file})...")
        subprocess.run(["dvc", "pull", dvc_file], check=True)
    except subprocess.CalledProcessError:
        print(f"DVC pull failed for {dvc_file}.")
        # We'll still check if the folder exists locally afterwards

    # Read the folder path tracked by this DVC file
    folder_path = None
    with open(dvc_file, "r") as f:
        for line in f:
            if line.strip().startswith("path:") or line.strip().startswith("outs:"):
                # Usually the tracked folder is on the next line
                folder_path = next(f).split(":")[-1].strip()
                break

    if not folder_path:
        raise ValueError(f"Could not determine tracked path from {dvc_file}")

    if not os.path.isdir(folder_path):
        raise RuntimeError(
            f"No valid dataset folder found at {folder_path} "
            "(DVC pull may have failed or local folder missing)."
        )

    print(f"✅ Dataset ready at {folder_path}")
    return folder_path


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

    # Repo root
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dvc_file = os.path.join(REPO_ROOT, "data.dvc")

    # Pull DVC data first
    dataset_path = pull_dvc_data(dvc_file)

    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)

    # Split into train/val/test
    train_length = int(0.8 * len(dataset))
    val_length = int(0.9 * len(dataset))
    indices = np.random.choice(len(dataset), len(dataset), replace=False)

    train_dataset = Subset(dataset, indices[:train_length])
    validation_dataset = Subset(dataset, indices[train_length:val_length])
    test_dataset = Subset(dataset, indices[val_length:])

    # DataLoaders
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=train_batch_size, persistent_workers=True)
    val_loader = DataLoader(validation_dataset, num_workers=num_workers, batch_size=train_batch_size, persistent_workers=True)
    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=train_batch_size, persistent_workers=True)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(validation_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
