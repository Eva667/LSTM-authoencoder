import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional
import pandas as pd

def compute_reconstruction_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray
) -> Dict[str, float]:
    """
    Вычисляет метрики качества восстановления.
    
    Args:
        X_true: исходные данные (n_samples, n_timesteps, n_features)
        X_pred: восстановленные данные
        
    Returns:
        Словарь с метриками
    """
    # Сглаживаем для вычисления метрик
    n_samples = X_true.shape[0]
    X_true_flat = X_true.reshape(n_samples, -1)
    X_pred_flat = X_pred.reshape(n_samples, -1)
    
    # Покомпонентные метрики
    mse_per_sample = np.mean((X_true_flat - X_pred_flat) ** 2, axis=1)
    mae_per_sample = np.mean(np.abs(X_true_flat - X_pred_flat), axis=1)
    
    # Общие метрики по всем данным
    mse = np.mean(mse_per_sample)
    mae = np.mean(mae_per_sample)
    rmse = np.sqrt(mse)
    
    # R² коэффициент детерминации
    ss_res = np.sum((X_true_flat - X_pred_flat) ** 2)
    ss_tot = np.sum((X_true_flat - np.mean(X_true_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Пиковая ошибка
    max_error = np.max(np.abs(X_true_flat - X_pred_flat))
    
    # Относительная ошибка (MAPE)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((X_true_flat - X_pred_flat) / (X_true_flat + 1e-10))) * 100
        mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'max_error': float(max_error),
        'mape': float(mape),
        'mse_per_sample_std': float(np.std(mse_per_sample)),
        'mae_per_sample_std': float(np.std(mae_per_sample))
    }
    
    return metrics


def compute_reconstruction_metrics_per_feature(
    X_true: np.ndarray,
    X_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Вычисляет метрики качества восстановления для каждого признака отдельно.
    
    Args:
        X_true: исходные данные (n_samples, n_timesteps, n_features)
        X_pred: восстановленные данные
        
    Returns:
        Словарь с метриками для каждого признака
    """
    n_samples, n_timesteps, n_features = X_true.shape
    
    mse_per_feature = []
    mae_per_feature = []
    r2_per_feature = []
    
    for f in range(n_features):
        X_true_f = X_true[:, :, f].reshape(n_samples, -1)
        X_pred_f = X_pred[:, :, f].reshape(n_samples, -1)
        
        mse_f = np.mean((X_true_f - X_pred_f) ** 2)
        mae_f = np.mean(np.abs(X_true_f - X_pred_f))
        
        # R^2 для признака
        ss_res = np.sum((X_true_f - X_pred_f) ** 2)
        ss_tot = np.sum((X_true_f - np.mean(X_true_f)) ** 2)
        r2_f = 1 - (ss_res / (ss_tot + 1e-10))
        
        mse_per_feature.append(mse_f)
        mae_per_feature.append(mae_f)
        r2_per_feature.append(r2_f)
    
    return {
        'mse_per_feature': np.array(mse_per_feature),
        'mae_per_feature': np.array(mae_per_feature),
        'r2_per_feature': np.array(r2_per_feature)
    }


def compute_correlation_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray
) -> Dict[str, float]:
    """
    Вычисляет корреляцию между истинными и восстановленными значениями.
    
    Args:
        X_true: исходные данные
        X_pred: восстановленные данные
        
    Returns:
        Словарь с корреляционными метриками
    """
    n_samples = X_true.shape[0]
    X_true_flat = X_true.reshape(n_samples, -1)
    X_pred_flat = X_pred.reshape(n_samples, -1)
    
    # Покорреляция Пирсона для каждого сэмпла
    correlations = []
    for i in range(n_samples):
        corr, _ = pearsonr(X_true_flat[i], X_pred_flat[i])
        if not np.isnan(corr):
            correlations.append(corr)
    
    mean_correlation = np.mean(correlations) if correlations else 0.0
    
    # Корреляция по всем данным
    all_true = X_true_flat.flatten()
    all_pred = X_pred_flat.flatten()
    global_corr, _ = pearsonr(all_true, all_pred)
    
    return {
        'mean_sample_correlation': float(mean_correlation),
        'global_correlation': float(global_corr if not np.isnan(global_corr) else 0.0),
        'correlation_std': float(np.std(correlations)) if correlations else 0.0
    }


def compute_temporal_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Вычисляет метрики, зависящие от времени (для каждого временного шага).
    
    Args:
        X_true: исходные данные
        X_pred: восстановленные данные
        
    Returns:
        Словарь с временными метриками
    """
    n_samples, n_timesteps, n_features = X_true.shape
    
    mse_per_timestep = np.zeros(n_timesteps)
    mae_per_timestep = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        X_true_t = X_true[:, t, :].reshape(n_samples, -1)
        X_pred_t = X_pred[:, t, :].reshape(n_samples, -1)
        
        mse_per_timestep[t] = np.mean((X_true_t - X_pred_t) ** 2)
        mae_per_timestep[t] = np.mean(np.abs(X_true_t - X_pred_t))
    
    return {
        'mse_per_timestep': mse_per_timestep,
        'mae_per_timestep': mae_per_timestep
    }


def compute_all_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    X_latent: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Вычисляет все метрики для полной оценки.
    
    Args:
        X_true: исходные данные
        X_pred: восстановленные данные
        X_latent: латентные представления (опционально)
        true_labels: истинные метки кластеров (опционально)
        cluster_labels: предсказанные метки кластеров (опционально)
        
    Returns:
        Словарь со всеми метриками
    """
    metrics = {}
    
    # Базовые метрики восстановления
    metrics['reconstruction'] = compute_reconstruction_metrics(X_true, X_pred)
    
    # Метрики по признакам
    metrics['per_feature'] = compute_reconstruction_metrics_per_feature(X_true, X_pred)
    
    # Корреляционные метрики
    metrics['correlation'] = compute_correlation_metrics(X_true, X_pred)
    
    # Временные метрики
    metrics['temporal'] = compute_temporal_metrics(X_true, X_pred)
    
    # Если есть латентные представления, добавляем статистику по ним
    if X_latent is not None:
        metrics['latent'] = {
            'mean': float(np.mean(X_latent)),
            'std': float(np.std(X_latent)),
            'min': float(np.min(X_latent)),
            'max': float(np.max(X_latent)),
            'shape': X_latent.shape
        }
    
    # Если есть метки кластеров, добавляем метрики кластеризации
    if cluster_labels is not None:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        n_unique = len(np.unique(cluster_labels))
        metrics['clustering'] = {
            'n_clusters': n_unique,
            'n_noise': int(np.sum(cluster_labels == -1)) if -1 in cluster_labels else 0
        }
        
        if n_unique > 1 and X_latent is not None:
            metrics['clustering']['silhouette'] = float(silhouette_score(X_latent, cluster_labels))
            metrics['clustering']['davies_bouldin'] = float(davies_bouldin_score(X_latent, cluster_labels))
            metrics['clustering']['calinski_harabasz'] = float(calinski_harabasz_score(X_latent, cluster_labels))
        
        # Внешние метрики, если есть истинные метки
        if true_labels is not None and n_unique > 1:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            metrics['clustering']['ari'] = float(adjusted_rand_score(true_labels, cluster_labels))
            metrics['clustering']['nmi'] = float(normalized_mutual_info_score(true_labels, cluster_labels))
    
    return metrics


def format_metrics_table(metrics: Dict[str, any]) -> pd.DataFrame:
    """
    Форматирует метрики в виде таблицы для отчета.
    
    Args:
        metrics: словарь с метриками
        
    Returns:
        DataFrame с метриками
    """
    rows = []
    
    # Основные метрики восстановления
    if 'reconstruction' in metrics:
        row = {'Тип': 'Восстановление'}
        row.update(metrics['reconstruction'])
        rows.append(row)
    
    # Метрики кластеризации
    if 'clustering' in metrics:
        row = {'Тип': 'Кластеризация'}
        row.update(metrics['clustering'])
        rows.append(row)
    
    # Корреляция
    if 'correlation' in metrics:
        row = {'Тип': 'Корреляция'}
        row.update(metrics['correlation'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_metrics_summary(metrics: Dict[str, any]):
    """
    Красиво выводит сводку метрик в консоль.
    """
    print("=" * 60)
    print("СВОДКА МЕТРИК")
    print("=" * 60)
    
    if 'reconstruction' in metrics:
        print("\n📊 МЕТРИКИ ВОССТАНОВЛЕНИЯ:")
        for key, value in metrics['reconstruction'].items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.6f}")
    
    if 'clustering' in metrics:
        print("\n📈 МЕТРИКИ КЛАСТЕРИЗАЦИИ:")
        for key, value in metrics['clustering'].items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")
    
    if 'correlation' in metrics:
        print("\n🔗 КОРРЕЛЯЦИЯ:")
        for key, value in metrics['correlation'].items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
    
    print("\n" + "=" * 60)


# Пример использования
if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_timesteps, n_features = 100, 50, 5
    
    X_true = np.random.randn(n_samples, n_timesteps, n_features)
    X_pred = X_true + np.random.randn(n_samples, n_timesteps, n_features) * 0.1
    metrics = compute_all_metrics(X_true, X_pred)
    print_metrics_summary(metrics)

    df = format_metrics_table(metrics)
    print("\nТаблица метрик:")
    print(df)