"""
Методы для определения оптимального количества кластеров.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class find_optimal_k_methods:
    """
    Класс с методами для определения оптимального количества кластеров.
    """
    
    @staticmethod
    def elbow_method(X: np.ndarray, max_k: int = 15) -> list:
        """
        Метод локтя для определения оптимального количества кластеров.
        
        Args:
            X: Нормализованные данные
            max_k: Максимальное количество кластеров
        
        Returns:
            list: Список значений искажений для разных k
        """
        distortions = []
        K = range(1, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
        
        return distortions
    
    @staticmethod
    def silhouette_analysis(X: np.ndarray, max_k: int = 15) -> list:
        """
        Силуэтный анализ для определения оптимального количества кластеров.
        
        Args:
            X: Нормализованные данные
            max_k: Максимальное количество кластеров
        
        Returns:
            list: Список значений силуэтных коэффициентов для разных k
        """
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        K = range(2, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        return silhouette_scores
    
    @staticmethod
    def generate_reference_dataset(data: np.ndarray) -> np.ndarray:
        """
        Генерирует референсное множество, равномерно распределённое в bounding box исходных данных.
        
        Args:
            data: Исходные данные формы (n_samples, n_features)
        
        Returns:
            np.ndarray: Референсное множество
        """
        n_samples, n_features = data.shape
        ref_data = np.empty_like(data)
        
        for j in range(n_features):
            min_j = data[:, j].min()
            max_j = data[:, j].max()
            ref_data[:, j] = np.random.uniform(min_j, max_j, size=n_samples)
        
        return ref_data
    
    @staticmethod
    def compute_gap_statistic(data: np.ndarray, max_k: int = 10, n_refs: int = 20) -> tuple:
        """
        Вычисление Gap Statistic для определения оптимального количества кластеров.
        
        Args:
            data: Нормализованные данные
            max_k: Максимальное количество кластеров
            n_refs: Количество референсных наборов
        
        Returns:
            tuple: (список Gap Statistic, список SSE)
        """
        gaps = []
        sses = []
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            sse = kmeans.inertia_
            sses.append(sse)
            
            ref_sses = []
            for _ in range(n_refs):
                ref_data = find_optimal_k_methods.generate_reference_dataset(data)
                ref_kmeans = KMeans(n_clusters=k, random_state=42)
                ref_kmeans.fit(ref_data)
                ref_sses.append(ref_kmeans.inertia_)
            
            gap = np.log(np.mean(ref_sses)) - np.log(sse)
            gaps.append(gap)
        
        return gaps, sses