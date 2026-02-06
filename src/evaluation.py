"""
Функции для оценки качества кластеризации.
"""

import torch
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.cluster import KMeans
from typing import Tuple, Dict


def find_optimal_k_elbow(distortions: list) -> int:
    """
    Нахождение оптимального количества кластеров методом локтя.
    
    Args:
        distortions: Список значений искажений для разных k
    
    Returns:
        int: Оптимальное количество кластеров
    """
    # Простая эвристика: ищем точку максимальной кривизны
    if len(distortions) < 3:
        return 2
    
    # Вычисляем вторую производную
    second_deriv = []
    for i in range(1, len(distortions) - 1):
        d1 = distortions[i] - distortions[i-1]
        d2 = distortions[i+1] - distortions[i]
        second_deriv.append(d2 - d1)
    
    # Находим точку максимальной кривизны
    if second_deriv:
        optimal_idx = np.argmax(np.abs(second_deriv)) + 1
        return optimal_idx + 1
    
    return 5


def find_optimal_k_silhouette(silhouette_scores: list) -> int:
    """
    Нахождение оптимального количества кластеров силуэтным анализом.
    
    Args:
        silhouette_scores: Список значений силуэтных коэффициентов
    
    Returns:
        int: Оптимальное количество кластеров
    """
    if not silhouette_scores:
        return 2
    
    # Находим k с максимальным силуэтным коэффициентом
    optimal_idx = np.argmax(silhouette_scores)
    return optimal_idx + 2  # +2 потому что начинаем с k=2


def evaluate_clustering(
    data: np.ndarray,
    labels: np.ndarray,
    method_name: str = "Clustering"
) -> Dict[str, float]:
    """
    Оценка качества кластеризации с помощью метрик.
    
    Args:
        data: Данные (нормализованные)
        labels: Метки кластеров
        method_name: Название метода для вывода
    
    Returns:
        dict: Словарь с метриками
    """
    metrics = {}
    
    # Силуэтный коэффициент
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(data, labels)
    else:
        metrics['silhouette'] = 0.0
    
    print(f"\n=== МЕТРИКИ КАЧЕСТВА: {method_name} ===")
    print(f"Силуэтный коэффициент: {metrics['silhouette']:.4f}")
    
    return metrics


def calculate_metrics(
    kmeans_labels: np.ndarray,
    dec_labels: np.ndarray,
    data: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Сравнение метрик кластеризации для K-means и DEC.
    
    Args:
        kmeans_labels: Метки кластеров от K-means
        dec_labels: Метки кластеров от DEC
        data: Нормализованные данные
    
    Returns:
        dict: Словарь с метриками для обоих методов
    """
    metrics = {}
    
    # Метрики для K-means
    metrics['kmeans'] = evaluate_clustering(data, kmeans_labels, "K-Means")
    
    # Метрики для DEC
    metrics['dec'] = evaluate_clustering(data, dec_labels, "DEC")
    
    # Сравнительные метрики (только если оба метода имеют одинаковое количество кластеров)
    if len(np.unique(kmeans_labels)) == len(np.unique(dec_labels)):
        homogeneity = homogeneity_score(kmeans_labels, dec_labels)
        completeness = completeness_score(kmeans_labels, dec_labels)
        v_measure = v_measure_score(kmeans_labels, dec_labels)
        
        metrics['comparison'] = {
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure
        }
        
        print(f"\n=== СРАВНЕНИЕ МЕТОДОВ ===")
        print(f"Homogeneity: {homogeneity:.4f}")
        print(f"Completeness: {completeness:.4f}")
        print(f"V-Measure: {v_measure:.4f}")
    
    return metrics


def analyze_cluster_statistics(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    method_name: str
) -> None:
    """
    Анализ статистики по кластерам.
    
    Args:
        data: Исходные данные (не нормализованные)
        labels: Метки кластеров
        feature_names: Названия признаков
        method_name: Название метода
    """
    print(f"\n=== СТАТИСТИКА ПО КЛАСТЕРАМ: {method_name} ===")
    
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_data = data[mask]
        
        print(f"\nКластер {cluster_id}:")
        print(f"  Количество точек: {len(cluster_data)}")
        
        for idx, feature in enumerate(feature_names):
            if idx < cluster_data.shape[1]:
                mean_val = cluster_data[:, idx].mean()
                std_val = cluster_data[:, idx].std()
                print(f"  {feature}: {mean_val:.2f} ± {std_val:.2f}")