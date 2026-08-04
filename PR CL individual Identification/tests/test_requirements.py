"""Test to verify all required packages are installed and importable."""
import sys
import importlib

REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "wandb": "wandb",
}


def test_python_version():
    """Verify Python version is 3.12 or compatible."""
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"


def test_torch_import():
    """Test that PyTorch can be imported."""
    import torch
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")


def test_torchvision_import():
    """Test that torchvision can be imported."""
    import torchvision
    assert torchvision.__version__ is not None
    print(f"torchvision version: {torchvision.__version__}")


def test_all_required_packages():
    """Test that all required packages can be imported."""
    failed_imports = []
    
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package_name} ({import_name}): {version}")
        except ImportError as e:
            failed_imports.append((package_name, str(e)))
            print(f"✗ {package_name} ({import_name}): FAILED - {e}")
    
    if failed_imports:
        error_msg = "Failed to import required packages:\n"
        for package, error in failed_imports:
            error_msg += f"  - {package}: {error}\n"
        raise ImportError(error_msg)
    
    assert len(failed_imports) == 0, f"Failed to import {len(failed_imports)} package(s)"
