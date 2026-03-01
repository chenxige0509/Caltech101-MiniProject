"""
Project config: paths, seed, device, and directory setup.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

# Project root (directory containing pyproject.toml / caltech-101 / src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
NUM_CLASSES = 101

# ImageNet normalization (used by torchvision models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create outputs/logs, outputs/figures, outputs/checkpoints, outputs/tables."""
    for name in ("logs", "figures", "checkpoints", "tables"):
        (PROJECT_ROOT / "outputs" / name).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# For notebook compatibility: expose as device
device = get_device()


def get_env_info() -> dict:
    """Return dict with Python, PyTorch, platform for logging."""
    import platform
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "?"),
        "platform": platform.platform(),
    }
