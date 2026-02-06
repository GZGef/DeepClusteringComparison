"""
Функции для визуализации результатов кластеризации.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple


def plot_elbow_method(distortions: list, save_path: str = 'elbow_method.png') -> None:
    """
    Визуализация метода локтя для определения оптимального количества кластеров.
    
    Args:
        distortions: Список значений искажений для разных k
        save_path: Путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    K = range(1, len(distortions) + 1)
    plt.plot(K, distortions, 'bx-', linewidth=2, markersize=8)
    plt.xlabel('Количество кластеров (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Искажение (Distortion)', fontsize=12, fontweight='bold')
    plt.title('Метод локтя для определения оптимального k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График метода локтя сохранен: {save_path}")


def plot_silhouette_analysis(silhouette_scores: list, save_path: str = 'silhouette_analysis.png') -> None:
    """
    Визуализация силуэтного анализа.
    
    Args:
        silhouette_scores: Список значений силуэтных коэффициентов
        save_path: Путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    K = range(2, len(silhouette_scores) + 2)
    plt.plot(K, silhouette_scores, 'bx-', linewidth=2, markersize=8)
    plt.xlabel('Количество кластеров (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Силуэтный коэффициент', fontsize=12, fontweight='bold')
    plt.title('Силуэтный анализ для определения оптимального k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График силуэтного анализа сохранен: {save_path}")


def plot_gap_statistic(gaps: list, save_path: str = 'gap_statistic.png') -> None:
    """
    Визуализация Gap Statistic.
    
    Args:
        gaps: Список значений Gap Statistic
        save_path: Путь для сохранения графика
    """
    plt.figure(figsize=(12, 6))
    K = range(1, len(gaps) + 1)
    plt.plot(K, gaps, 'bx-', linewidth=2, markersize=8)
    plt.xlabel('Количество кластеров (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Gap Statistic', fontsize=12, fontweight='bold')
    plt.title('Gap Statistic для определения оптимального k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График Gap Statistic сохранен: {save_path}")


def plot_results(
    data: np.ndarray,
    dec_embeddings: np.ndarray,
    dec_labels: np.ndarray,
    cluster_centers: np.ndarray,
    feature_names: list,
    save_path: str = 'dec_results.png'
) -> None:
    """
    Визуализация результатов DEC кластеризации.
    
    Args:
        data: Исходные данные
        dec_embeddings: Эмбеддинги DEC
        dec_labels: Метки кластеров DEC
        cluster_centers: Центры кластеров
        feature_names: Названия признаков
        save_path: Путь для сохранения графика
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. DEC Latent Space
    plt.subplot(2, 3, 1)
    if dec_embeddings.shape[1] >= 2:
        scatter = plt.scatter(
            dec_embeddings[:, 0], dec_embeddings[:, 1],
            c=dec_labels, cmap='viridis', alpha=0.7, s=60,
            edgecolors='w', linewidth=0.5
        )
        plt.scatter(
            cluster_centers[:, 0], cluster_centers[:, 1],
            c='red', marker='X', s=300, linewidth=3,
            edgecolors='black', label='Центры кластеров'
        )
        plt.title('DEC Latent Space\n(место кластеризации)', fontsize=12, fontweight='bold')
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.legend()
        plt.colorbar(scatter, label='Кластер')
        plt.grid(True, alpha=0.3)
    
    # 2. t-SNE всех признаков
    plt.subplot(2, 3, 2)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data)
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=dec_labels, cmap='viridis', alpha=0.7, s=60,
        edgecolors='w', linewidth=0.5
    )
    plt.title('t-SNE всех признаков\nс метками DEC', fontsize=12, fontweight='bold')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Кластер')
    plt.grid(True, alpha=0.3)
    
    # 3. PCA всех признаков
    plt.subplot(2, 3, 3)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(data)
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=dec_labels, cmap='viridis', alpha=0.7, s=60,
        edgecolors='w', linewidth=0.5
    )
    plt.title(
        f'PCA всех признаков\n(объяснено {pca.explained_variance_ratio_.sum()*100:.1f}% дисперсии)',
        fontsize=12, fontweight='bold'
    )
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Кластер')
    plt.grid(True, alpha=0.3)
    
    # 4-6. Pairplot для всех комбинаций признаков
    feature_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_titles = [
        f'{feature_names[0]} vs {feature_names[1]}',
        f'{feature_names[0]} vs {feature_names[2]}',
        f'{feature_names[1]} vs {feature_names[2]}'
    ]
    
    for i, (idx1, idx2) in enumerate(feature_pairs):
        plt.subplot(2, 3, i + 4)
        scatter = plt.scatter(
            data[:, idx1], data[:, idx2],
            c=dec_labels, cmap='viridis', alpha=0.7, s=60,
            edgecolors='w', linewidth=0.5
        )
        plt.title(pair_titles[i], fontsize=11, fontweight='bold')
        plt.xlabel(feature_names[idx1])
        plt.ylabel(feature_names[idx2])
        plt.colorbar(scatter, label='Кластер')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График результатов DEC сохранен: {save_path}")


def plot_comparison(
    data: np.ndarray,
    kmeans_labels: np.ndarray,
    dec_labels: np.ndarray,
    feature_names: list,
    save_path: str = 'kmeans_vs_dec_comparison.png'
) -> None:
    """
    Визуализация сравнения K-means и DEC.
    
    Args:
        data: Исходные данные
        kmeans_labels: Метки кластеров K-means
        dec_labels: Метки кластеров DEC
        feature_names: Названия признаков
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # K-means
    scatter1 = axes[0].scatter(
        data[:, 0], data[:, 1],
        c=kmeans_labels, cmap='viridis', alpha=0.7, s=50,
        edgecolors='w', linewidth=0.5
    )
    axes[0].set_title('K-Means Clustering', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(feature_names[0], fontsize=12)
    axes[0].set_ylabel(feature_names[1], fontsize=12)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Кластер')
    
    # DEC
    scatter2 = axes[1].scatter(
        data[:, 0], data[:, 1],
        c=dec_labels, cmap='viridis', alpha=0.7, s=50,
        edgecolors='w', linewidth=0.5
    )
    axes[1].set_title('DEC Clustering', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(feature_names[0], fontsize=12)
    axes[1].set_ylabel(feature_names[1], fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Кластер')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График сравнения методов сохранен: {save_path}")


def plot_distributions(
    data: np.ndarray,
    kmeans_labels: np.ndarray,
    dec_labels: np.ndarray,
    feature_names: list,
    n_clusters: int,
    save_path: str = 'cluster_distributions.png'
) -> None:
    """
    Визуализация распределения по кластерам.
    
    Args:
        data: Исходные данные
        kmeans_labels: Метки кластеров K-means
        dec_labels: Метки кластеров DEC
        feature_names: Названия признаков
        n_clusters: Количество кластеров
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Распределение по первому признаку
    for cluster_id in range(n_clusters):
        axes[0, 0].hist(
            data[kmeans_labels == cluster_id, 0],
            alpha=0.7,
            label=f'Кластер {cluster_id}',
            bins=15,
            density=True
        )
    axes[0, 0].set_title(f'K-Means: Распределение по {feature_names[0]}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel(feature_names[0], fontsize=10)
    axes[0, 0].set_ylabel('Плотность', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for cluster_id in range(n_clusters):
        axes[0, 1].hist(
            data[dec_labels == cluster_id, 0],
            alpha=0.7,
            label=f'Кластер {cluster_id}',
            bins=15,
            density=True
        )
    axes[0, 1].set_title(f'DEC: Распределение по {feature_names[0]}', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel(feature_names[0], fontsize=10)
    axes[0, 1].set_ylabel('Плотность', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Распределение по второму признаку
    for cluster_id in range(n_clusters):
        axes[1, 0].hist(
            data[kmeans_labels == cluster_id, 1],
            alpha=0.7,
            label=f'Кластер {cluster_id}',
            bins=15,
            density=True
        )
    axes[1, 0].set_title(f'K-Means: Распределение по {feature_names[1]}', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel(feature_names[1], fontsize=10)
    axes[1, 0].set_ylabel('Плотность', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for cluster_id in range(n_clusters):
        axes[1, 1].hist(
            data[dec_labels == cluster_id, 1],
            alpha=0.7,
            label=f'Кластер {cluster_id}',
            bins=15,
            density=True
        )
    axes[1, 1].set_title(f'DEC: Распределение по {feature_names[1]}', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel(feature_names[1], fontsize=10)
    axes[1, 1].set_ylabel('Плотность', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График распределений сохранен: {save_path}")


def plot_training_history(
    ae_loss_history: list,
    dec_loss_history: list,
    dec_shift_history: list,
    save_path: str = 'training_history.png'
) -> None:
    """
    Визуализация истории обучения.
    
    Args:
        ae_loss_history: История потерь автоэнкодера
        dec_loss_history: История потерь DEC
        dec_shift_history: История смещения центров DEC
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Потери автоэнкодера
    axes[0].plot(ae_loss_history, linewidth=2)
    axes[0].set_title('Потери автоэнкодера', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Эпоха', fontsize=10)
    axes[0].set_ylabel('MSE Loss', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Потери DEC
    axes[1].plot(dec_loss_history, linewidth=2)
    axes[1].set_title('Потери DEC', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Эпоха', fontsize=10)
    axes[1].set_ylabel('KL Loss', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Смещение центров
    axes[2].plot(dec_shift_history, linewidth=2)
    axes[2].set_title('Смещение центров кластеров', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Эпоха', fontsize=10)
    axes[2].set_ylabel('Смещение', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График истории обучения сохранен: {save_path}")