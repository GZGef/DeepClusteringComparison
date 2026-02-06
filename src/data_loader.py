"""
Загрузка и предобработка данных для кластеризации.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os


def load_and_preprocess_data(url: str = None, file_path: str = 'Mall_Customers.csv') -> tuple:
    """
    Загружает и предобрабатывает датасет Mall_Customers.
    
    Args:
        url: URL для загрузки датасета (опционально)
        file_path: Путь к локальному файлу датасета
    
    Returns:
        tuple: (нормализованные данные, исходный DataFrame, названия признаков)
    """
    # Загрузка данных
    if url and not os.path.exists(file_path):
        print(f"Загрузка датасета с {url}...")
        import requests
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Датасет успешно загружен: {file_path}")
    
    print(f"Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path, sep=',')
    
    # Удаление ненужного столбца с id
    df = df.drop(columns=["CustomerID"])
    
    # Преобразование категориального признака в числовой
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    df["Gender"] = df["Gender_Male"].astype(int)
    df = df.drop(columns=["Gender_Male"])
    
    # Вывод информации о данных
    print("\n=== ИНФОРМАЦИЯ О ДАННЫХ ===")
    print(f"Размер датасета: {df.shape}")
    print(f"\nПризнаки: {list(df.columns)}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nСтатистика по признакам:")
    print(df.describe())
    
    # Проверка пропущенных значений
    print("\n=== ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ ===")
    missing_values = df.isna().sum()
    if missing_values.sum() == 0:
        print("Пропущенных значений не обнаружено")
    else:
        print(missing_values)
    
    # Проверка выбросов
    print("\n=== ПРОВЕРКА ВЫБРОСОВ ===")
    for col in df.columns:
        data_col = df[col]
        
        # Метод IQR
        Q1 = data_col.quantile(0.25)
        Q3 = data_col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound_iqr = Q1 - 1.5 * IQR
        upper_bound_iqr = Q3 + 1.5 * IQR
        outliers_iqr_mask = (data_col < lower_bound_iqr) | (data_col > upper_bound_iqr)
        num_outliers_iqr = outliers_iqr_mask.sum()
        
        # Метод Z-score
        mean_val = data_col.mean()
        std_val = data_col.std()
        z_scores = np.abs((data_col - mean_val) / std_val)
        outliers_z_mask = z_scores > 3
        num_outliers_z = outliers_z_mask.sum()
        
        print(f"\nПризнак: {col}")
        print(f"  IQR: {IQR:.2f}, Границы: ({lower_bound_iqr:.2f}, {upper_bound_iqr:.2f})")
        print(f"  Выбросов (IQR): {num_outliers_iqr}")
        print(f"  Выбросов (Z-score > 3): {num_outliers_z}")
    
    # Нормализация данных
    print("\n=== НОРМАЛИЗАЦИЯ ДАННЫХ ===")
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    print(f"Нормализованные данные (первые 5 строк):")
    print(df_normalized[:5])
    
    feature_names = list(df.columns)
    
    return df_normalized, df, feature_names


def plot_data_distribution(df: pd.DataFrame, save_path: str = 'data_distribution.png') -> None:
    """
    Визуализация распределения данных.
    
    Args:
        df: Исходный DataFrame
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        if idx < len(axes):
            axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Распределение: {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Частота')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График распределения данных сохранен: {save_path}")