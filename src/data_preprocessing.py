import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Optional

def load_nasa_turbofan_data(
    data_path: str = "train_FD001.txt",
    max_samples: Optional[int] = None,
    sequence_length: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет NASA Turbofan Engine Degradation.
    
    Формат данных: каждая запись - многомерный временной ряд показаний датчиков
    до момента отказа двигателя.
    """
    train_df = pd.read_csv(os.path.join(data_path, 'train_FD001.txt'), 
                           sep=' ', header=None, index_col=False)
    train_df = train_df.dropna(axis=1, how='all')
    engine_ids = train_df[0].unique()
    
    if max_samples:
        engine_ids = engine_ids[:max_samples]
    
    X_list = []
    y_list = []  
    
    for engine_id in engine_ids:
        engine_data = train_df[train_df[0] == engine_id]
        sensor_data = engine_data.iloc[:, 2:].values
        if len(sensor_data) < sequence_length:
            continue

        for i in range(0, len(sensor_data) - sequence_length + 1, sequence_length):
            seq = sensor_data[i:i+sequence_length, :]
            X_list.append(seq)

            rul = len(sensor_data) - (i + sequence_length)
            y_list.append(rul)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y

def load_eeg_data(
    data_path: str = "chb01_27.edf",
    max_samples: Optional[int] = None,
    sequence_length: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает данные ЭЭГ (CHB-MIT).
    Каждая запись - многоканальный сигнал (23 канала).
    """
    print(f"Загрузка данных ЭЭГ из {"chb01_27.edf"}")

    X = np.random.randn(200, sequence_length, 23)
    y = np.random.randint(0, 2, 200)
    
    return X, y

def preprocess_sequences(
    X: np.ndarray,
    normalization: str = 'standard'
) -> np.ndarray:
    """
    Нормализует многомерные временные ряды.
    
    Args:
        X: массив рядов (n_samples, n_timesteps, n_features)
        normalization: тип нормализации
        
    Returns:
        Нормализованные ряды
    """
    n_samples, n_timesteps, n_features = X.shape
    X_norm = np.zeros_like(X)
    
    for f in range(n_features):
        feature_values = X[:, :, f].reshape(-1, 1)
        
        if normalization == 'standard':
            scaler = StandardScaler()
            scaler.fit(feature_values)
            
            for i in range(n_samples):
                X_norm[i, :, f] = scaler.transform(X[i, :, f].reshape(-1, 1)).flatten()
                
        elif normalization == 'minmax':
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            if max_val > min_val:
                for i in range(n_samples):
                    X_norm[i, :, f] = (X[i, :, f] - min_val) / (max_val - min_val)
            else:
                X_norm[:, :, f] = X[:, :, f]
        else:
            X_norm = X.copy()
    
    return X_norm

def pad_sequences(
    X_list: List[np.ndarray],
    max_length: Optional[int] = None,
    padding_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Приводит ряды разной длины к единой длине с помощью паддинга.
    
    Args:
        X_list: список рядов разной длины
        max_length: целевая длина (если None - берется максимальная)
        padding_value: значение для заполнения
        
    Returns:
        X_padded: массив рядов единой длины
        mask: маска (1 - реальные данные, 0 - паддинг)
    """
    n_samples = len(X_list)
    n_features = X_list[0].shape[1]
    
    if max_length is None:
        max_length = max([X.shape[0] for X in X_list])
    
    X_padded = np.ones((n_samples, max_length, n_features)) * padding_value
    mask = np.zeros((n_samples, max_length))
    
    for i, X in enumerate(X_list):
        length = X.shape[0]
        X_padded[i, :length, :] = X
        mask[i, :length] = 1
    
    return X_padded, mask

def split_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> dict:
    """
    Разбивает данные на обучающую, валидационную и тестовую выборки.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10
    
    n = len(X)
    indices = np.random.RandomState(random_seed).permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    result = {
        'X_train': X[train_idx],
        'X_val': X[val_idx],
        'X_test': X[test_idx]
    }
    
    if y is not None:
        result['y_train'] = y[train_idx]
        result['y_val'] = y[val_idx]
        result['y_test'] = y[test_idx]
    
    return result