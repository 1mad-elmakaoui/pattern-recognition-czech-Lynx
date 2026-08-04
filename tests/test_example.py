"""Example test file showing pytest best practices."""
import pytest
import torch


def test_simple_assertion():
    """Simple test with assertion."""
    assert 1 + 1 == 2


def test_torch_tensor_creation():
    """Test PyTorch tensor creation."""
    tensor = torch.zeros(3, 3)
    assert tensor.shape == (3, 3)
    assert tensor.sum() == 0.0


@pytest.fixture
def sample_tensor():
    """Example fixture for reusable test data."""
    return torch.randn(3, 3)


def test_with_fixture(sample_tensor):
    """Test using a fixture."""
    assert sample_tensor.shape == (3, 3)
    assert sample_tensor.dtype == torch.float32


@pytest.mark.parametrize("size", [(2, 2), (3, 3), (4, 4)])
def test_parametrized(size):
    """Example of parametrized test (runs multiple times with different inputs)."""
    tensor = torch.zeros(size)
    assert tensor.shape == size

