import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import os


def get_dataloaders(seed=0, num_workers=9, train_batch_size=64):
    np.random.seed(seed)

    data_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Sizing
            transforms.Grayscale(),  # Making Black and White
            transforms.ToTensor(),  # Convert torch tensor
        ],
    )
    path = os.getcwd()

    path = os.path.join(path, "data")

    dataset = datasets.ImageFolder(root=path, transform=data_transform)

    # split into indices
    train_length = int(0.8 * len(dataset))
    val_length = int(0.9 * len(dataset))

    indices = np.random.choice(len(dataset), len(dataset), replace=False)

    train_dataset = Subset(dataset, indices[:train_length])
    validation_dataset = Subset(dataset, indices[train_length:val_length])
    test_dataset = Subset(dataset, indices[val_length:])

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

    get_data_splits()