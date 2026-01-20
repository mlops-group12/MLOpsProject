"""
Data Loading Module (DVC-first, cross-platform support)
"""

import os
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import tempfile
import subprocess


def pull_dvc_temp(dvc_file: str) -> str:
    """
    Pull a DVC-tracked dataset into a temporary folder and return the path.
    """
    tmp_dir = tempfile.mkdtemp()

    print(f"Pulling dataset via DVC ({dvc_file}) into temporary folder {tmp_dir}...")
    subprocess.run([
        "dvc", "pull", dvc_file, "--out", tmp_dir
    ], check=True)

    return tmp_dir


def get_dataloaders(
    seed: int = 0,
    num_workers: int = 4,
    train_batch_size: int = 64,
) -> tuple:
    """
    Create train, validation, and test DataLoaders.

    Workflow:
    1. Attempt to pull dataset via DVC (ensures versioning and reproducibility).
    2. If DVC fails, fallback to local 'data/' folder in repo root.
    """

    np.random.seed(seed)

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # -------------------------
    # Determine dataset path
    # -------------------------
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    local_data_path = os.path.join(REPO_ROOT, "data")

    try:
        # Priority: use DVC to pull the dataset
        dataset_path = pull_dvc_temp(os.path.join(REPO_ROOT, "data.dvc"))
    except Exception as e:
        print(f"DVC pull failed: {e}")
        if os.path.isdir(local_data_path):
            print(f"Falling back to local data folder: {local_data_path}")
            dataset_path = local_data_path
        else:
            raise RuntimeError("No valid dataset path found (DVC failed and local folder missing).")

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = datasets.ImageFolder(root=dataset_path, transform=data_transform)

    # Split into train/validation/test
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)  # 10% validation
    n_test = n_total - n_train - n_val

    indices = np.random.permutation(n_total)
    train_dataset = Subset(dataset, indices[:n_train])
    val_dataset = Subset(dataset, indices[n_train:n_train + n_val])
    test_dataset = Subset(dataset, indices[n_train + n_val:])

    # -------------------------
    # Create DataLoaders
    # -------------------------
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, num_workers=num_workers, persistent_workers=True)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
