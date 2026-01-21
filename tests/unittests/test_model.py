import torch
from mlopsproject.model import CNN
import pytest


class TestModel:
    number_of_classes = 5
    input_size = (64, 64)

    @pytest.mark.parametrize("batch_size", [16, 32, 64])
    def test_forward(self, batch_size):
        model = CNN()
        x = torch.randn(batch_size, 1, *self.input_size)
        y = model(x)
        assert y.shape == (batch_size, self.number_of_classes)

    def test_training_step(self):
        model = CNN()
        x = torch.randn(1, 1, *self.input_size)
        y = torch.randn(1, 5)
        loss = model.training_step((x, y), batch_idx=0)
        assert loss is not None
        assert torch.is_tensor(loss)

    def test_validation_step(self):
        model = CNN()

        x = torch.randn(1, 1, *self.input_size)
        y = torch.randn(1, 5)

        out = model.validation_step((x, y), batch_idx=0)
        assert out is None

    def test_test_step(self):
        model = CNN()
        x = torch.randn(1, 1, *self.input_size)
        y = torch.randn(1, 5)

        model.on_test_start()

        y = model.test_step((x, y), batch_idx=0)

        assert y is None

    def test_configure_optimizers(self):
        model = CNN()
        optimizer = model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)
