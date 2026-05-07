import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from typing import Optional, List

def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Строит графики обучения автоэнкодера."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Потери
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Потери при обучении')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    if 'mae' in history and 'val_mae' in history:
        axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Средняя абсолютная ошибка')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    plt.show()

def plot_latent_space(
    X_latent: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    title: str = 'Латентное пространство',
    save_path: Optional[str] = None
):
    """Визуализирует латентное пространство с помощью PCA, t-SNE или UMAP."""
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_latent)
        explained_var = reducer.explained_variance_ratio_.sum()
        title += f' (PCA, explained var: {explained_var:.2f})'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = reducer.fit_transform(X_latent)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_latent)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    plt.figure(figsize=(10, 8))
    
    # Раскраска по кластерам
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                          c=labels, cmap='tab10', 
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label='Кластер')
    plt.title(title, fontsize=14)
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    plt.show()

def plot_reconstruction_examples(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    n_examples: int = 4,
    feature_idx: int = 0,
    save_path: Optional[str] = None
):
    """Показывает примеры исходных и восстановленных рядов."""
    n_samples = min(len(X_true), n_examples)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        true_series = X_true[i, :, feature_idx]
        pred_series = X_pred[i, :, feature_idx]
        
        time_steps = np.arange(len(true_series))
        
        ax.plot(time_steps, true_series, 'b-', label='Исходный', linewidth=2, alpha=0.8)
        ax.plot(time_steps, pred_series, 'r--', label='Восстановленный', linewidth=2, alpha=0.8)
        
        mse = np.mean((true_series - pred_series) ** 2)
        ax.set_title(f'Пример {i+1}, MSE = {mse:.4f}', fontsize=12)
        ax.set_xlabel('Время')
        ax.set_ylabel(f'Признак {feature_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    plt.show()

def plot_reconstruction_error_distribution(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """Строит распределение ошибок восстановления."""
    n_samples = X_true.shape[0]
    X_true_flat = X_true.reshape(n_samples, -1)
    X_pred_flat = X_pred.reshape(n_samples, -1)
    
    # MSE для каждого сэмпла
    mse_per_sample = np.mean((X_true_flat - X_pred_flat) ** 2, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограмма
    axes[0].hist(mse_per_sample, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(mse_per_sample), color='red', linestyle='--', 
                    label=f'Среднее: {np.mean(mse_per_sample):.4f}')
    axes[0].axvline(np.median(mse_per_sample), color='green', linestyle='--',
                    label=f'Медиана: {np.median(mse_per_sample):.4f}')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Распределение ошибок восстановления')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(mse_per_sample, vert=True)
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Box plot ошибок')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    plt.show()

def plot_cluster_profiles(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    n_features_to_plot: int = 3,
    save_path: Optional[str] = None
):
    """Строит профили кластеров (средние ряды для каждого кластера)."""
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  # Исключаем шум
    
    n_clusters = len(unique_clusters)
    n_features = min(n_features_to_plot, X.shape[2])
    
    fig, axes = plt.subplots(n_features, n_clusters, figsize=(4*n_clusters, 3*n_features))
    
    if n_features == 1:
        axes = [axes]
    if n_clusters == 1:
        axes = [[ax] for ax in axes]
    
    for f in range(n_features):
        for i, cluster in enumerate(unique_clusters):
            ax = axes[f][i] if n_features > 1 else axes[i]
            
            cluster_data = X[cluster_labels == cluster, :, f]
            mean_series = np.mean(cluster_data, axis=0)
            std_series = np.std(cluster_data, axis=0)
            
            time_steps = np.arange(len(mean_series))
            
            ax.plot(time_steps, mean_series, 'b-', linewidth=2)
            ax.fill_between(time_steps, mean_series - std_series, mean_series + std_series,
                           alpha=0.3, color='blue')
            
            ax.set_title(f'Кластер {cluster}, признак {f}')
            ax.set_xlabel('Время')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    
    plt.show()