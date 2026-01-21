"""
Data Drift Evaluation Module

This module evaluates robustness to data drift by comparing feature distributions
between the training dataset and a simulated drifted dataset.
"""

import os
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report

from mlopsproject.data import get_dataloaders

try:
    from google.cloud import storage
except Exception:
    storage = None


def extract_features(dataloader, max_batches: int):
    """
    Extract simple image statistics from a DataLoader.
    """
    records = []

    for batch_idx, (images, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        # robust to GPU tensors
        images = images.detach().cpu().numpy()

        for img in images:
            img = img.squeeze()

            brightness = float(img.mean())
            contrast = float(img.std())

            gx, gy = np.gradient(img)
            sharpness = float(np.mean(np.sqrt(gx**2 + gy**2)))

            records.append(
                {
                    "brightness": brightness,
                    "contrast": contrast,
                    "sharpness": sharpness,
                }
            )

    return pd.DataFrame.from_records(records)


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """
    Upload a file to GCS.
    """
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage is not installed or not available. "
            "Install it or disable drift.save_to_gcs."
        )

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)


@hydra.main(
    config_path="../../configs",
    config_name="drift_config",
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Run an offline data drift analysis comparing:
    - Training data 
    - CelebA 
    """

    train_loader, _, _ = get_dataloaders(
        gcs_bucket=cfg.gcs.bucket,
        gcs_folder=cfg.gcs.data_folder,
        num_workers=2,
    )

    reference_df = extract_features(train_loader, max_batches=cfg.drift.max_batches)
    print(f"Reference samples: {len(reference_df)}")

    celeba_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    celeba_dataset = datasets.ImageFolder(
        root=cfg.drift.celeba_root,
        transform=celeba_transform,
    )


    celeba_loader = DataLoader(
        celeba_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
    )

    current_df = extract_features(celeba_loader, max_batches=cfg.drift.max_batches)
    print(f"Current samples: {len(current_df)}")

    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_df, current_data=current_df)

    # Use config for output
    os.makedirs(cfg.drift.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.drift.output_dir, f"{cfg.drift.report_name}.html")
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

    if cfg.drift.save_to_gcs:
        if not cfg.gcs.bucket:
            raise ValueError("cfg.gcs.bucket is empty but drift.save_to_gcs=true")

        gcs_folder = cfg.drift.gcs_report_folder.strip("/")
        gcs_report_path = f"{gcs_folder}/{cfg.drift.report_name}.html"

        upload_to_gcs(output_path, cfg.gcs.bucket, gcs_report_path)
        print(f"Uploaded drift report to gs://{cfg.gcs.bucket}/{gcs_report_path}")


if __name__ == "__main__":
    main()
