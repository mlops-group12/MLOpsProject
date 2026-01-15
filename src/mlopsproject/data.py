"""
Data Loading Module (cross-platform GCS support)
"""

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import os
from google.cloud import storage
import tempfile
import shutil


def download_gcs_folder(bucket_name, gcs_folder, local_folder):
    """
    Download all files from a GCS folder to a local folder.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_folder)
    for blob in blobs:
        # Skip folders
        if blob.name.endswith("/"):
            continue
        # Compute relative path
        rel_path = os.path.relpath(blob.name, gcs_folder)
        local_file_path = os.path.join(local_folder, rel_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)


def get_dataloaders(
    seed=0, num_workers=4, train_batch_size=64,
    local_path=None, gcs_bucket=None, gcs_folder=None
):
    """
    Create train, validation, and test DataLoaders.

    Supports local directories or GCS folders.
    """

    np.random.seed(seed)

    data_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    # Determine dataset path
    if local_path and os.path.isdir(local_path):
        path = local_path
        print(f"Local folder '{local_path}' exists, using it. Skipping GCS download.")
    elif gcs_bucket and gcs_folder:
        # Use temporary folder for GCS download
        tmp_dir = tempfile.mkdtemp()
        print(f"Downloading data from GCS bucket {gcs_bucket}/{gcs_folder} to {tmp_dir} ...")
        download_gcs_folder(gcs_bucket, gcs_folder, tmp_dir)
        path = tmp_dir
    else:
        raise ValueError("No valid dataset path found")

    print("Using dataset path:", path)
    dataset = datasets.ImageFolder(root=path, transform=data_transform)

    # Split indices
    train_length = int(0.8 * len(dataset))
    val_length = int(0.9 * len(dataset))
    indices = np.random.choice(len(dataset), len(dataset), replace=False)

    train_dataset = Subset(dataset, indices[:train_length])
    validation_dataset = Subset(dataset, indices[train_length:val_length])
    test_dataset = Subset(dataset, indices[val_length:])

    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(validation_dataset))
    print("Number of test samples:", len(test_dataset))

    # Convert to DataLoaders
    train_loader = DataLoader(
        train_dataset, num_workers=num_workers,
        batch_size=train_batch_size, persistent_workers=True
    )
    val_loader = DataLoader(
        validation_dataset, num_workers=num_workers,
        batch_size=train_batch_size, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, num_workers=num_workers,
        batch_size=train_batch_size, persistent_workers=True
    )

    return train_loader, val_loader, test_loader
