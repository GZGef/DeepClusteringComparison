"""
Утилиты для проекта кластеризации.
"""

import torch
import numpy as np
import random


def set_seed(seed: int = 42) -> None:
    """
    Устанавливает seed для воспроизводимости результатов.
    
    Args:
        seed: Значение seed для инициализации генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Определяет доступное вычислительное устройство (GPU или CPU).
    
    Returns:
        torch.device: Вычислительное устройство
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    return device