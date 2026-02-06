"""
Основной скрипт для сравнения методов кластеризации: K-means и Deep Embedded Clustering (DEC).

Этот скрипт выполняет:
1. Загрузку и предобработку данных Mall_Customers
2. Определение оптимального количества кластеров
3. Обучение K-means
4. Обучение DEC (автоэнкодер + кластеризация)
5. Оценку качества кластеризации
6. Визуализацию результатов
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchinfo import summary

# Импорт модулей проекта
from src.utils import set_seed, get_device
from src.data_loader import load_and_preprocess_data, plot_data_distribution
from src.models import Autoencoder, DEC
from src.training import train_autoencoder, train_dec
from src.evaluation import (
    evaluate_clustering,
    calculate_metrics,
    analyze_cluster_statistics
)
from src.visualization import (
    plot_elbow_method,
    plot_silhouette_analysis,
    plot_gap_statistic,
    plot_results,
    plot_comparison,
    plot_distributions,
    plot_training_history
)
from src.clustering_methods import find_optimal_k_methods


def main():
    """
    Основная функция для выполнения всего пайплайна кластеризации.
    """
    print("=" * 80)
    print("СРАВНЕНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ: K-MEANS VS DEEP EMBEDDED CLUSTERING")
    print("=" * 80)
    
    # 1. Установка seed для воспроизводимости
    print("\n=== 1. УСТАНОВКА SEED ===")
    set_seed(42)
    
    # 2. Определение вычислительного устройства
    print("\n=== 2. ОПРЕДЕЛЕНИЕ УСТРОЙСТВА ===")
    device = get_device()
    
    # 3. Загрузка и предобработка данных
    print("\n=== 3. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ ===")
    df_normalized, df_original, feature_names = load_and_preprocess_data(
        url='https://storage.yandexcloud.net/google-colab-bucket/Mall_Customers.csv',
        file_path='Mall_Customers.csv'
    )
    
    # Визуализация распределения данных
    plot_data_distribution(df_original, save_path='data_distribution.png')
    
    # 4. Определение оптимального количества кластеров
    print("\n=== 4. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ ===")
    
    # Метод локтя
    distortions = find_optimal_k_methods.elbow_method(df_normalized, max_k=15)
    plot_elbow_method(distortions, save_path='elbow_method.png')
    
    # Силуэтный анализ
    silhouette_scores = find_optimal_k_methods.silhouette_analysis(df_normalized, max_k=15)
    plot_silhouette_analysis(silhouette_scores, save_path='silhouette_analysis.png')
    
    # Gap Statistic
    gaps, _ = find_optimal_k_methods.compute_gap_statistic(df_normalized, max_k=15)
    plot_gap_statistic(gaps, save_path='gap_statistic.png')
    
    # Определяем оптимальное количество кластеров
    n_clusters = 11  # Выбрано на основе анализа
    print(f"\nОптимальное количество кластеров: {n_clusters}")
    
    # 5. Создание DataLoader
    print("\n=== 5. СОЗДАНИЕ DATALOADER ===")
    df_tensor = torch.tensor(df_normalized, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(df_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )
    
    # 6. Создание и обучение моделей
    print("\n=== 6. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ ===")
    
    # Создание автоэнкодера
    autoencoder = Autoencoder(
        input_dim=4,
        hidden_dim=16,
        latent_dim=2
    ).to(device)
    
    print("\nАрхитектура автоэнкодера:")
    summary(
        autoencoder,
        input_size=(16, 4),
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )
    
    # Предобучение автоэнкодера
    ae_loss, ae_loss_history = train_autoencoder(
        autoencoder=autoencoder,
        dataloader=dataloader,
        device=device,
        epochs=500,
        learning_rate=1e-3
    )
    
    # Создание DEC модели
    dec_model = DEC(
        autoencoder=autoencoder,
        n_clusters=n_clusters,
        latent_dim=2,
        alpha=1.0
    ).to(device)
    
    print("\nАрхитектура DEC модели:")
    summary(
        dec_model,
        input_size=(16, 4),
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )
    
    # Инициализация центров кластеров
    dec_model.initialize_clusters(dataloader, device)
    
    # Обучение DEC
    dec_loss, dec_loss_history, dec_shift_history = train_dec(
        dec_model=dec_model,
        dataloader=dataloader,
        device=device,
        epochs=250,
        learning_rate=1e-2
    )
    
    # 7. Получение предсказаний DEC
    print("\n=== 7. ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ DEC ===")
    dec_model.eval()
    with torch.no_grad():
        df_tensor_device = df_tensor.to(device)
        Z_final, _, Q_final = dec_model(df_tensor_device)
        Z_final = Z_final.cpu().numpy()
        Q_final = Q_final.cpu().numpy()
        dec_labels = np.argmax(Q_final, axis=1)
    
    # 8. Обучение K-means
    print("\n=== 8. ОБУЧЕНИЕ K-MEANS ===")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(df_normalized)
    
    # 9. Оценка качества кластеризации
    print("\n=== 9. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ ===")
    
    # Метрики для K-means
    kmeans_metrics = evaluate_clustering(df_normalized, kmeans_labels, "K-Means")
    
    # Метрики для DEC (на эмбеддингах)
    dec_metrics = evaluate_clustering(Z_final, dec_labels, "DEC")
    
    # Сравнительные метрики
    metrics = calculate_metrics(kmeans_labels, dec_labels, df_normalized)
    
    # 10. Анализ статистики по кластерам
    print("\n=== 10. АНАЛИЗ СТАТИСТИКИ ПО КЛАСТЕРАМ ===")
    analyze_cluster_statistics(
        df_original.values,
        kmeans_labels,
        feature_names,
        "K-Means"
    )
    analyze_cluster_statistics(
        df_original.values,
        dec_labels,
        feature_names,
        "DEC"
    )
    
    # 11. Визуализация результатов
    print("\n=== 11. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ===")
    
    # Результаты DEC
    plot_results(
        data=df_normalized,
        dec_embeddings=Z_final,
        dec_labels=dec_labels,
        cluster_centers=dec_model.cluster_centers.detach().cpu().numpy(),
        feature_names=feature_names,
        save_path='dec_results.png'
    )
    
    # Сравнение K-means и DEC
    plot_comparison(
        data=df_normalized,
        kmeans_labels=kmeans_labels,
        dec_labels=dec_labels,
        feature_names=feature_names,
        save_path='kmeans_vs_dec_comparison.png'
    )
    
    # Распределение по кластерам
    plot_distributions(
        data=df_normalized,
        kmeans_labels=kmeans_labels,
        dec_labels=dec_labels,
        feature_names=feature_names,
        n_clusters=n_clusters,
        save_path='cluster_distributions.png'
    )
    
    # История обучения
    plot_training_history(
        ae_loss_history=ae_loss_history,
        dec_loss_history=dec_loss_history,
        dec_shift_history=dec_shift_history,
        save_path='training_history.png'
    )
    
    # 12. Вывод итогов
    print("\n" + "=" * 80)
    print("ИТОГИ СРАВНЕНИЯ МЕТОДОВ")
    print("=" * 80)
    
    print(f"\nСилуэтный коэффициент K-Means: {kmeans_metrics['silhouette']:.4f}")
    print(f"Силуэтный коэффициент DEC: {dec_metrics['silhouette']:.4f}")
    
    if 'comparison' in metrics:
        print(f"\nHomogeneity: {metrics['comparison']['homogeneity']:.4f}")
        print(f"Completeness: {metrics['comparison']['completeness']:.4f}")
        print(f"V-Measure: {metrics['comparison']['v_measure']:.4f}")
    
    print("\n" + "=" * 80)
    print("ВСЕ ГРАФИКИ СОХРАНЕНЫ В ТЕКУЩЕЙ ДИРЕКТОРИИ")
    print("=" * 80)


if __name__ == "__main__":
    main()