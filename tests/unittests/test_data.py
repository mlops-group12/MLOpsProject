from torch.utils.data import Dataset
import pytest
from mlopsproject.data import get_dataloaders
import os

path = os.getcwd()

path = os.path.join(path, "data")


@pytest.fixture
def fixture_get_dataloaders():
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders()
    return train_dataloader, validation_dataloader, test_dataloader


@pytest.fixture
def fixture_get_datasets(fixture_get_dataloaders):
    train_dataloader, validation_dataloader, test_dataloader = fixture_get_dataloaders

    return (
        train_dataloader.dataset,
        validation_dataloader.dataset,
        test_dataloader.dataset,
    )


@pytest.mark.skipif(not os.path.exists(path), reason="Processed data not found")
def test_my_dataset(fixture_get_datasets):
    train_dataset = fixture_get_datasets[0]
    val_dataset = fixture_get_datasets[1]
    test_dataset = fixture_get_datasets[2]

    assert isinstance(train_dataset, Dataset)
    assert isinstance(val_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    assert len(train_dataset) == 47279, f"Train dataset size is {len(train_dataset)} expected 47279"
    assert len(test_dataset) == 5910, f"Test dataset size is {len(test_dataset)} expected 5910"

    print("Dataset length test passed.")


def test_my_dataloader_batch_shape(fixture_get_dataloaders):
    x, y = next(iter(fixture_get_dataloaders[0]))

    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    assert x.shape[1:] == (1, 64, 64), f"Expected (*, 1, 64, 64), got {x.shape}"

    print("Dataloader batch shape test passed.")
