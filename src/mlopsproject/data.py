"""
Data Loading Module (cross-platform GCS support)
"""

import os
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import tempfile
from google.cloud import storage


def download_gcs_folder(bucket_name, gcs_folder, local_folder):
    """
    Download all files from a GCS folder to a local folder.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_folder)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path = os.path.relpath(blob.name, gcs_folder)
        local_file_path = os.path.join(local_folder, rel_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)


def get_dataloaders(
    seed=0,
    num_workers=4,
    train_batch_size=64,
    gcs_bucket=None,
    gcs_folder=None
):
    """
    Create train, validation, and test DataLoaders.

    This function prioritizes the local repo-root 'data/' folder.
    If the folder does not exist, it falls back to downloading from GCS.
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
    # Repo root = two levels up from this file
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    local_data_path = os.path.join(REPO_ROOT, "data")

    if os.path.isdir(local_data_path):
        dataset_path = local_data_path
        print(f"Using local data folder: {dataset_path}")
    elif gcs_bucket and gcs_folder:
        tmp_dir = tempfile.mkdtemp()
        print(f"Local data not found, downloading from GCS bucket {gcs_bucket}/{gcs_folder} to {tmp_dir} ...")
        download_gcs_folder(gcs_bucket, gcs_folder, tmp_dir)
        dataset_path = tmp_dir
    else:
        raise ValueError("No valid dataset path found (local or GCS)")

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
