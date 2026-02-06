"""
Пакет для сравнения методов кластеризации: K-means и Deep Embedded Clustering (DEC).
"""

from .data_loader import load_and_preprocess_data
from .models import Autoencoder, DEC
from .training import train_autoencoder, train_dec
from .evaluation import evaluate_clustering, calculate_metrics
from .visualization import plot_results, plot_comparison, plot_distributions
from .utils import set_seed, get_device

__all__ = [
    'load_and_preprocess_data',
    'Autoencoder',
    'DEC',
    'train_autoencoder',
    'train_dec',
    'evaluate_clustering',
    'calculate_metrics',
    'plot_results',
    'plot_comparison',
    'plot_distributions',
    'set_seed',
    'get_device',
]