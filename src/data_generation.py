import numpy as np
from typing import Tuple, Optional

def generate_synthetic_data(
    n_samples: int = 1000,
    n_timesteps: int = 50,
    n_features: int = 5,
    n_classes: int = 4,
    noise_level: float = 0.1,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует синтетические многомерные временные ряды с известными классами.
    
    Классы:
    0: Гармонические колебания (синусоиды с разными частотами)
    1: Затухающие колебания
    2: Пилообразные сигналы
    3: Случайный шум (baseline)
    
    Args:
        n_samples: количество рядов
        n_timesteps: длина каждого ряда
        n_features: количество признаков
        n_classes: количество классов
        noise_level: уровень шума
        random_seed: для воспроизводимости
        
    Returns:
        X: массив рядов (n_samples, n_timesteps, n_features)
        y: метки классов (n_samples,)
    """
    np.random.seed(random_seed)
    
    X = np.zeros((n_samples, n_timesteps, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        class_type = i % n_classes
        y[i] = class_type
        
        t = np.linspace(0, 4*np.pi, n_timesteps)
        
        if class_type == 0:  # Гармонические колебания
            # Разные частоты для разных признаков
            base_signal = np.zeros((n_timesteps, n_features))
            for f in range(n_features):
                freq = 1.0 + 0.5 * f
                base_signal[:, f] = np.sin(freq * t + np.random.randn() * 0.5)
                
        elif class_type == 1:  # Затухающие колебания
            decay = np.exp(-t / 5)
            base_signal = np.zeros((n_timesteps, n_features))
            for f in range(n_features):
                freq = 1.0 + 0.3 * f
                base_signal[:, f] = np.sin(freq * t) * decay
                
        elif class_type == 2:  # Пилообразные сигналы
            base_signal = np.zeros((n_timesteps, n_features))
            sawtooth = (t % 2) / 2
            for f in range(n_features):
                base_signal[:, f] = sawtooth + 0.1 * f
                
        else:  # Случайный шум
            base_signal = np.zeros((n_timesteps, n_features))
            # Без базового сигнала, только шум
            
        # Добавляем шум
        noise = np.random.randn(n_timesteps, n_features) * noise_level
        X[i] = base_signal + noise
        
        # Нормализуем каждый ряд индивидуально (опционально)
        for f in range(n_features):
            mean = np.mean(X[i, :, f])
            std = np.std(X[i, :, f])
            if std > 0:
                X[i, :, f] = (X[i, :, f] - mean) / std
    
    return X, y

def generate_varying_length_data(
    n_samples: int = 500,
    min_length: int = 30,
    max_length: int = 100,
    n_features: int = 5,
    n_classes: int = 3
) -> list:
    """
    Генерирует ряды разной длины для тестирования устойчивости.
    
    Returns:
        Список рядов разной длины
    """
    np.random.seed(42)
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        length = np.random.randint(min_length, max_length)
        class_type = i % n_classes
        
        t = np.linspace(0, 4*np.pi, length)
        X = np.zeros((length, n_features))
        
        if class_type == 0:
            for f in range(n_features):
                X[:, f] = np.sin(t + f)
        elif class_type == 1:
            for f in range(n_features):
                X[:, f] = np.cos(t * 0.5 + f)
        else:
            for f in range(n_features):
                X[:, f] = np.random.randn(length) * 0.5
                
        X_list.append(X)
        y_list.append(class_type)
    
    return X_list, np.array(y_list)

def save_synthetic_data(X, y, filepath: str):
    """Сохраняет синтетические данные в файл"""
    np.savez(filepath, X=X, y=y)
    print(f"Данные сохранены в {filepath}, форма X: {X.shape}")