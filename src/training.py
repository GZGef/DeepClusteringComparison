"""
Функции для обучения моделей: автоэнкодера и DEC.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time
from typing import Tuple

from .models import Autoencoder, DEC


def train_autoencoder(
    autoencoder: Autoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    verbose: bool = True
) -> Tuple[float, list]:
    """
    Предобучение автоэнкодера.
    
    Args:
        autoencoder: Модель автоэнкодера
        dataloader: DataLoader с данными
        device: Вычислительное устройство
        epochs: Количество эпох
        learning_rate: Скорость обучения
        verbose: Вывод информации о процессе обучения
    
    Returns:
        tuple: (средняя потеря, история потерь)
    """
    print("\n=== ПРЕДОБУЧЕНИЕ АВТОЭНКОДЕРА ===")
    
    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    scheduler = StepLR(
        optimizer,
        step_size=100,
        gamma=0.5
    )
    
    autoencoder.train()
    loss_history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            
            # Forward pass
            z, x_recon = autoencoder(x)
            loss = criterion(x_recon, x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(
                f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
                f'Time: {(time.time()-start_time):.2f} сек'
            )
    
    print(f'Предобучение завершено. Средняя потеря: {avg_loss:.4f}')
    return avg_loss, loss_history


def update_cluster_centers(dec_model: DEC, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, float]:
    """
    Обновление центров кластеров на основе текущих эмбеддингов.
    
    Args:
        dec_model: Модель DEC
        dataloader: DataLoader с данными
    
    Returns:
        tuple: (новые центры кластеров, среднее смещение)
    """
    device = next(dec_model.parameters()).device
    dec_model.eval()
    
    with torch.no_grad():
        all_z = []
        all_q = []
        
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            
            z, _, q = dec_model(x)
            all_z.append(z)
            all_q.append(q)
        
        Z = torch.cat(all_z, dim=0)
        Q = torch.cat(all_q, dim=0)
        
        prev_centroids = dec_model.cluster_centers.clone()
        
        new_centroids = torch.zeros_like(dec_model.cluster_centers)
        for j in range(dec_model.n_clusters):
            weights = Q[:, j]
            weighted_sum = torch.sum(weights.unsqueeze(1) * Z, dim=0)
            weight_sum = torch.sum(weights)
            new_centroids[j] = weighted_sum / (weight_sum + 1e-10)
        
        centroid_shift = torch.norm(new_centroids - prev_centroids, dim=1).mean().item()
        
        return new_centroids, centroid_shift


def train_dec(
    dec_model: DEC,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 250,
    learning_rate: float = 1e-2,
    epsilon: float = 1e-5,
    verbose: bool = True
) -> Tuple[float, list, list]:
    """
    Обучение модели DEC.
    
    Args:
        dec_model: Модель DEC
        dataloader: DataLoader с данными
        device: Вычислительное устройство
        epochs: Количество эпох
        learning_rate: Скорость обучения
        epsilon: Порог сходимости
        verbose: Вывод информации о процессе обучения
    
    Returns:
        tuple: (средняя потеря, история потерь, история смещения центров)
    """
    print("\n=== ОБУЧЕНИЕ DEC МОДЕЛИ ===")
    
    optimizer = optim.SGD(
        dec_model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    criterion = nn.KLDivLoss(reduction='batchmean')
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-5
    )
    
    dec_model.train()
    loss_history = []
    shift_history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            
            # Forward pass
            z, x_recon, q = dec_model(x)
            
            # Target distribution
            with torch.no_grad():
                p = dec_model.target_distribution(q)
            
            # KL divergence loss
            loss = criterion(q.log(), p)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # Обновление центров кластеров
        with torch.no_grad():
            new_centroids, centroid_shift = update_cluster_centers(dec_model, dataloader)
            dec_model.cluster_centers.copy_(new_centroids)
        
        shift_history.append(centroid_shift)
        scheduler.step(centroid_shift)
        
        if verbose and (epoch + 1) % 25 == 0:
            print(
                f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
                f'Centroid shift: {centroid_shift:.6f}, Time: {(time.time()-start_time):.2f} сек'
            )
        
        # Проверка сходимости
        if centroid_shift < epsilon:
            print(f"\nСходимость достигнута на эпохе {epoch+1}! Остановка обучения.")
            break
    
    print(f'Обучение DEC завершено. Средняя потеря: {avg_loss:.4f}')
    return avg_loss, loss_history, shift_history