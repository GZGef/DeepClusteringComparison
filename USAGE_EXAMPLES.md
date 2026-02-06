# Примеры использования проекта

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Запуск проекта

```bash
python main.py
```

### 3. Ожидаемый результат

Проект выполнит следующие шаги:
1. Загрузка и предобработка данных Mall_Customers
2. Определение оптимального количества кластеров (3 метода)
3. Предобучение автоэнкодера (500 эпох)
4. Обучение DEC модели (до 250 эпох)
5. Обучение K-means
6. Оценка качества кластеризации
7. Визуализация результатов (8 графиков)

## 📊 Примеры вывода

### Метрики качества

```
=== МЕТРИКИ КАЧЕСТВА: K-Means ===
Силуэтный коэффициент: 0.4523

=== МЕТРИКИ КАЧЕСТВА: DEC ===
Силуэтный коэффициент: 0.5217

=== СРАВНЕНИЕ МЕТОДОВ ===
Homogeneity: 0.6542
Completeness: 0.6789
V-Measure: 0.6663
```

### Статистика по кластерам (K-Means)

```
=== СТАТИСТИКА ПО КЛАСТЕРАМ: K-Means ===

Кластер 0:
  Количество точек: 45
  Средний Annual Income: $45.23k
  Средний Age: 32.5 лет
  Средний Spending Score: 52.3

Кластер 1:
  Количество точек: 38
  Средний Annual Income: $78.45k
  Средний Age: 45.2 лет
  Средний Spending Score: 42.1
```

## 🎯 Примеры модификации проекта

### Изменение количества кластеров

```python
# В main.py измените n_clusters
n_clusters = 5  # Вместо 11
```

### Изменение параметров обучения

```python
# В main.py измените параметры
ae_loss, ae_loss_history = train_autoencoder(
    autoencoder=autoencoder,
    dataloader=dataloader,
    device=device,
    epochs=300,  # Меньше эпох
    learning_rate=1e-4  # Другая скорость обучения
)

dec_loss, dec_loss_history, dec_shift_history = train_dec(
    dec_model=dec_model,
    dataloader=dataloader,
    device=device,
    epochs=150,  # Меньше эпох
    learning_rate=1e-3  # Другая скорость обучения
)
```

### Использование другого датасета

```python
# В src/data_loader.py измените функцию load_and_preprocess_data
def load_and_preprocess_data(url: str = None, file_path: str = 'your_dataset.csv') -> tuple:
    # Загрузка вашего датасета
    df = pd.read_csv(file_path, sep=',')
    
    # Предобработка
    # ...
    
    return df_normalized, df, feature_names
```

### Добавление новых метрик

```python
# В src/evaluation.py добавьте новые метрики
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def calculate_metrics(kmeans_labels, dec_labels, data):
    metrics = {}
    
    # Существующие метрики
    metrics['kmeans'] = evaluate_clustering(data, kmeans_labels, "K-Means")
    metrics['dec'] = evaluate_clustering(data, dec_labels, "DEC")
    
    # Новые метрики
    if len(np.unique(kmeans_labels)) == len(np.unique(dec_labels)):
        ari = adjusted_rand_score(kmeans_labels, dec_labels)
        nmi = normalized_mutual_info_score(kmeans_labels, dec_labels)
        
        metrics['comparison']['adjusted_rand_index'] = ari
        metrics['comparison']['normalized_mutual_info'] = nmi
    
    return metrics
```

## 🔧 Отладка и диагностика

### Проверка доступности GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Уменьшение времени выполнения

Для быстрого тестирования можно уменьшить количество эпох:

```python
# В main.py
ae_loss, ae_loss_history = train_autoencoder(
    # ...
    epochs=50,  # Вместо 500
)

dec_loss, dec_loss_history, dec_shift_history = train_dec(
    # ...
    epochs=25,  # Вместо 250
)
```

### Сохранение моделей

```python
# В main.py после обучения
import torch

# Сохранение автоэнкодера
torch.save(autoencoder.state_dict(), 'autoencoder.pth')

# Сохранение DEC модели
torch.save({
    'model_state_dict': dec_model.state_dict(),
    'cluster_centers': dec_model.cluster_centers,
}, 'dec_model.pth')

# Загрузка моделей
autoencoder = Autoencoder(input_dim=4, hidden_dim=16, latent_dim=2)
autoencoder.load_state_dict(torch.load('autoencoder.pth'))

dec_model = DEC(autoencoder, n_clusters=11, latent_dim=2, alpha=1.0)
checkpoint = torch.load('dec_model.pth')
dec_model.load_state_dict(checkpoint['model_state_dict'])
dec_model.cluster_centers = checkpoint['cluster_centers']
```

## 📈 Сравнение с другими методами

### DBSCAN (для сравнения)

```python
from sklearn.cluster import DBSCAN

# Добавьте в main.py
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_normalized)

# Оценка (если есть шумовые точки)
unique_labels = np.unique(dbscan_labels)
if -1 in unique_labels:
    # Удаление шума для оценки
    mask = dbscan_labels != -1
    if np.sum(mask) > 1:
        dbscan_metrics = evaluate_clustering(
            df_normalized[mask], 
            dbscan_labels[mask], 
            "DBSCAN"
        )
```

### Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

# Добавьте в main.py
agg = AgglomerativeClustering(n_clusters=11)
agg_labels = agg.fit_predict(df_normalized)

agg_metrics = evaluate_clustering(df_normalized, agg_labels, "Agglomerative")
```

## 🎓 Образовательные примеры

### Визуализация эмбеддингов

```python
# Визуализация латентного пространства
import matplotlib.pyplot as plt

# Получение эмбеддингов
with torch.no_grad():
    z, _ = autoencoder(df_tensor.to(device))
    z = z.cpu().numpy()

# 2D визуализация
plt.figure(figsize=(10, 8))
plt.scatter(z[:, 0], z[:, 1], alpha=0.6, s=50)
plt.title('Latent Space Visualization')
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.grid(True, alpha=0.3)
plt.savefig('latent_space.png', dpi=300)
plt.show()
```

### Анализ ошибок

```python
# Анализ ошибок восстановления
with torch.no_grad():
    z, x_recon = autoencoder(df_tensor.to(device))
    errors = torch.mean((df_tensor.to(device) - x_recon) ** 2, dim=1).cpu().numpy()

# Визуализация ошибок
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.title('Распределение ошибок восстановления')
plt.xlabel('MSE Error')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)
plt.savefig('reconstruction_errors.png', dpi=300)
plt.show()
```

## 📚 Дополнительные ресурсы

### Полезные команды

```bash
# Установка дополнительных библиотек
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Проверка установки
python -c "import torch; print(torch.__version__)"
python -c "import sklearn; print(sklearn.__version__)"

# Запуск с отладкой
python -m pdb main.py
```

### Визуализация в Jupyter

```python
# В Jupyter Notebook
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Загрузка и отображение графиков
from IPython.display import Image
Image('elbow_method.png')
```

## 🎯 Чек-лист перед запуском

- [ ] Установлены все зависимости (`pip install -r requirements.txt`)
- [ ] Доступен датасет Mall_Customers.csv (или скрипт может его скачать)
- [ ] Достаточно свободного места на диске (для сохранения графиков)
- [ ] Python 3.8+ установлен
- [ ] PyTorch корректно установлен (проверка: `import torch; print(torch.cuda.is_available())`)
- [ ] Все файлы проекта на месте

## 🐛 Известные проблемы и решения

### Проблема: "No module named 'torchinfo'"

**Решение:**
```bash
pip install torchinfo
```

### Проблема: "CUDA out of memory"

**Решение:**
- Уменьшите batch_size в main.py
- Уменьшите количество эпох
- Используйте CPU: `device = torch.device('cpu')`

### Проблема: "Dataset not found"

**Решение:**
- Скачайте датасет вручную: `wget https://storage.yandexcloud.net/google-colab-bucket/Mall_Customers.csv`
- Или используйте локальный файл

### Проблема: "Matplotlib backend error"

**Решение:**
```python
import matplotlib
matplotlib.use('Agg')  # Добавьте в начало main.py
```

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте логи вывода в консоли
2. Убедитесь, что все зависимости установлены
3. Проверьте наличие датасета
4. Уменьшите параметры обучения для тестирования

## 🎓 Дальнейшее изучение

### Следующие шаги:
1. Изучите архитектуру DEC в статье "Unsupervised Deep Embedded Clustering"
2. Попробуйте применить метод к другим датасетам
3. Экспериментируйте с гиперпараметрами
4. Добавьте новые метрики оценки
5. Реализуйте другие глубокие методы кластеризации

### Рекомендуемая литература:
- "Deep Learning" by Ian Goodfellow (глубокое обучение)
- "Pattern Recognition and Machine Learning" by Christopher Bishop (кластеризация)
- "Unsupervised Deep Embedded Clustering" by Xie et al. (DEC)