"""Cố định seed cho toàn bộ pipeline (Reproducibility rubric)."""
from __future__ import annotations
import os, random
import numpy as np


def seed_everything(seed: int = 42) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Đánh đổi tốc độ lấy tính lặp lại — nêu rõ trong báo cáo.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
