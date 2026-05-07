import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union

class LatentSpaceClusterer:
    """
    Кластеризация латентных представлений.
    """
    
    def __init__(self, algorithm: str = 'agglomerative', **kwargs):
        """
        Args:
            algorithm: 'kmeans', 'agglomerative', 'dbscan', 'gmm'
            **kwargs: параметры алгоритма кластеризации
        """
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
        self.scaler = StandardScaler()
        
    def _create_model(self, n_clusters: Optional[int] = None):
        """Создает модель кластеризации."""
        if self.algorithm == 'kmeans':
            k = n_clusters if n_clusters else self.kwargs.get('n_clusters', 3)
            return KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=42,
                **{k: v for k, v in self.kwargs.items() if k != 'n_clusters'}
            )
        
        elif self.algorithm == 'agglomerative':
            k = n_clusters if n_clusters else self.kwargs.get('n_clusters', 3)
            return AgglomerativeClustering(
                n_clusters=k,
                linkage=self.kwargs.get('linkage', 'ward'),
                **{k: v for k, v in self.kwargs.items() if k not in ['n_clusters', 'linkage']}
            )
        
        elif self.algorithm == 'dbscan':
            return DBSCAN(
                eps=self.kwargs.get('eps', 0.5),
                min_samples=self.kwargs.get('min_samples', 5),
                **{k: v for k, v in self.kwargs.items() if k not in ['eps', 'min_samples']}
            )
        
        elif self.algorithm == 'gmm':
            k = n_clusters if n_clusters else self.kwargs.get('n_clusters', 3)
            return GaussianMixture(
                n_components=k,
                random_state=42,
                **{k: v for k, v in self.kwargs.items() if k != 'n_clusters'}
            )
        
        else:
            raise ValueError(f"Неизвестный алгоритм: {self.algorithm}")
    
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None):
        """
        Обучает модель кластеризации.
        """
        # Нормализуем данные
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = self._create_model(n_clusters)
        
        if self.algorithm == 'gmm':
            self.model.fit(X_scaled)
            self.labels_ = self.model.predict(X_scaled)
        else:
            self.labels_ = self.model.fit_predict(X_scaled)
        
        return self
    
    def fit_predict(self, X: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Обучает и возвращает метки кластеров.
        """
        self.fit(X, n_clusters)
        return self.labels_
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Возвращает центры кластеров (если поддерживается моделью).
        """
        if self.algorithm == 'kmeans':
            return self.model.cluster_centers_
        elif self.algorithm == 'gmm':
            return self.model.means_
        else:
            return None

class ClusterOptimizer:
    """
    Оптимизация числа кластеров.
    """
    
    @staticmethod
    def find_optimal_k(
        X: np.ndarray,
        algorithm: str = 'kmeans',
        k_range: range = range(2, 11),
        metric: str = 'silhouette',
        **kwargs
    ) -> Tuple[int, Dict[int, float]]:
        """
        Находит оптимальное число кластеров.
        
        Args:
            X: данные
            algorithm: алгоритм кластеризации
            k_range: диапазон для перебора
            metric: метрика ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
            optimal_k, scores_dict
        """
        scores = {}
        
        for k in k_range:
            clusterer = LatentSpaceClusterer(algorithm=algorithm, n_clusters=k, **kwargs)
            labels = clusterer.fit_predict(X)
            
            if metric == 'silhouette':
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                else:
                    score = -1
            elif metric == 'davies_bouldin':
                if len(np.unique(labels)) > 1:
                    score = -davies_bouldin_score(X, labels)  # Чем меньше, тем лучше, поэтому инвертируем
                else:
                    score = -np.inf
            elif metric == 'calinski_harabasz':
                if len(np.unique(labels)) > 1:
                    score = calinski_harabasz_score(X, labels)
                else:
                    score = -np.inf
            else:
                raise ValueError(f"Неизвестная метрика: {metric}")
            
            scores[k] = score
        
        optimal_k = max(scores, key=lambda k: scores[k])
        return optimal_k, scores

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Оценивает качество кластеризации.
    
    Args:
        X: исходные данные
        labels: предсказанные метки кластеров
        true_labels: истинные метки (если есть)
        
    Returns:
        Словарь с метриками
    """
    metrics = {}
    
    # Внутренние метрики
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(X, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    else:
        metrics['silhouette'] = -1
        metrics['davies_bouldin'] = np.inf
        metrics['calinski_harabasz'] = -np.inf
    
    # Внешние метрики (если есть истинные метки)
    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
    
    # Количество кластеров и шумовых точек
    metrics['n_clusters'] = len(np.unique(labels))
    metrics['n_noise'] = np.sum(labels == -1) if -1 in labels else 0
    
    return metrics

def compare_clustering_algorithms(
    X: np.ndarray,
    algorithms: List[str],
    n_clusters: int,
    true_labels: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Сравнивает различные алгоритмы кластеризации.
    """
    results = []
    
    for algo in algorithms:
        try:
            clusterer = LatentSpaceClusterer(algorithm=algo, n_clusters=n_clusters)
            labels = clusterer.fit_predict(X)
            metrics = evaluate_clustering(X, labels, true_labels)
            metrics['algorithm'] = algo
            results.append(metrics)
        except Exception as e:
            print(f"Ошибка при выполнении {algo}: {e}")
    
    return pd.DataFrame(results)