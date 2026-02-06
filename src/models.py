"""
Модели для кластеризации: Autoencoder и Deep Embedded Clustering (DEC).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class Autoencoder(nn.Module):
    """
    Автоэнкодер для предобучения эмбеддингов.
    
    Архитектура:
    - Энкодер: input_dim -> hidden_dim -> hidden_dim -> latent_dim
    - Декодер: latent_dim -> hidden_dim -> hidden_dim -> input_dim
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Инициализация автоэнкодера.
        
        Args:
            input_dim: Размерность входных данных
            hidden_dim: Размерность скрытого слоя
            latent_dim: Размерность пространства эмбеддингов
        """
        super().__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Инициализация весов с помощью Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Прямой проход через автоэнкодер.
        
        Args:
            x: Входной тензор
            
        Returns:
            tuple: (эмбеддинги, восстановленные данные)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Кодирование данных в эмбеддинги.
        
        Args:
            x: Входной тензор
            
        Returns:
            torch.Tensor: Эмбеддинги
        """
        return self.encoder(x)


class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC) - глубокий метод кластеризации.
    
    DEC использует автоэнкодер для получения эмбеддингов и оптимизирует
    распределение данных по кластерам с помощью soft assignment.
    
    Алгоритм работы:
    1. Предобучение автоэнкодера (инициализация эмбеддингов)
    2. Инициализация центров кластеров с помощью K-means
    3. Итеративная оптимизация:
       - Soft assignment: вычисление вероятностей принадлежности к кластерам
       - Target distribution: вычисление целевого распределения
       - Обновление весов: минимизация KL-дивергенции
       - Обновление центров кластеров
    """
    
    def __init__(self, autoencoder: Autoencoder, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        """
        Инициализация DEC модели.
        
        Args:
            autoencoder: Предобученный автоэнкодер
            n_clusters: Количество кластеров
            latent_dim: Размерность пространства эмбеддингов
            alpha: Параметр распределения Стьюдента
        """
        super().__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Центры кластеров (инициализируются позже)
        self.cluster_centers = nn.Parameter(
            torch.zeros(n_clusters, latent_dim),
            requires_grad=False
        )
    
    def initialize_clusters(self, data_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """
        Инициализация центров кластеров с помощью K-means.
        
        Args:
            data_loader: DataLoader с данными
            device: Вычислительное устройство
        """
        print("\n=== ИНИЦИАЛИЗАЦИЯ ЦЕНТРОВ КЛАСТЕРОВ ===")
        z_list = []
        self.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                z, _ = self.autoencoder(x)
                z_list.append(z.cpu())
        
        Z = torch.cat(z_list, dim=0).numpy()
        
        if len(Z) < self.n_clusters:
            raise ValueError(
                f"Недостаточно данных для кластеризации: {len(Z)} точек, "
                f"нужно минимум {self.n_clusters}"
            )
        
        print(f"Выполняется K-means на {len(Z)} точках...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(Z)
        
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32,
            device=device
        )
        self.cluster_centers.data = cluster_centers
        print(f"Центры кластеров инициализированы")
    
    def soft_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """
        Вычисление soft assignment (вероятностей принадлежности к кластерам).
        
        Формула: q_ij = (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2) / sum_j(...)
        
        Args:
            z: Эмбеддинги [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Матрица вероятностей [batch_size, n_clusters]
        """
        # Расстояние до центров кластеров
        dist = torch.cdist(z, self.cluster_centers, p=2)
        squared_dist = dist ** 2
        
        # Вычисление q_ij
        q = 1.0 / (1.0 + squared_dist / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2)
        
        # Нормализация
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Вычисление целевого распределения P.
        
        Формула: p_ij = q_ij^2 / sum_j(q_ij^2)
        
        Args:
            q: Матрица soft assignment [batch_size, n_clusters]
            
        Returns:
            torch.Tensor: Целевое распределение [batch_size, n_clusters]
        """
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        sum_weight = torch.sum(weight, dim=1, keepdim=True)
        return weight / sum_weight
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Прямой проход через DEC модель.
        
        Args:
            x: Входной тензор
            
        Returns:
            tuple: (эмбеддинги, восстановленные данные, soft assignment)
        """
        z, x_recon = self.autoencoder(x)
        q = self.soft_assignment(z)
        return z, x_recon, q