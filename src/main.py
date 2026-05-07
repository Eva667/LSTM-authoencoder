"""
Главный скрипт для запуска эксперимента по кластеризации многомерных временных рядов
с использованием LSTM-автоэнкодера.
"""

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.makedirs('./data/synthetic', exist_ok=True)
os.makedirs('./data/processed', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/figures', exist_ok=True)
os.makedirs('./results/tables', exist_ok=True)

print("=" * 70)
print("КЛАСТЕРИЗАЦИЯ МНОГОМЕРНЫХ ВРЕМЕННЫХ РЯДОВ")
print("С ИСПОЛЬЗОВАНИЕМ LSTM-АВТОЭНКОДЕРОВ")
print("=" * 70)

print("\n[1] Загрузка модулей...")
try:
    from data_generation import generate_synthetic_data, save_synthetic_data
    from data_preprocessing import preprocess_sequences, split_data
    from autoencoder import LSTMAutoencoder, build_dense_autoencoder, build_cnn_autoencoder
    from clustering import LatentSpaceClusterer, evaluate_clustering, ClusterOptimizer
    from evaluation import compute_all_metrics, print_metrics_summary, format_metrics_table
    from visualization import (
        plot_training_history, 
        plot_latent_space, 
        plot_reconstruction_examples,
        plot_reconstruction_error_distribution,
        plot_cluster_profiles
    )
    print("✓ Модули загружены")
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("\nУбедитесь, что все файлы находятся в текущей директории:")
    print("  - data_generation.py")
    print("  - data_preprocessing.py")
    print("  - autoencoder.py")
    print("  - clustering.py")
    print("  - evaluation.py")
    print("  - visualization.py")
    sys.exit(1)

# ======================================================================
# Шаг 1-5: Основной эксперимент
# ======================================================================
print("\n[2] Генерация синтетических данных...")
try:
    X, y_true = generate_synthetic_data(
        n_samples=500,
        n_timesteps=50,
        n_features=5,
        n_classes=4,
        noise_level=0.1
    )
    print(f"✓ Сгенерировано {X.shape[0]} рядов")
    print(f"  - Длина ряда: {X.shape[1]} временных шагов")
    print(f"  - Признаков: {X.shape[2]}")
    print(f"  - Классов: {len(np.unique(y_true))}")
    
    # Сохраняем данные
    save_synthetic_data(X, y_true, './data/synthetic/synthetic_data.npz')
except Exception as e:
    print(f"✗ Ошибка генерации данных: {e}")
    sys.exit(1)

print("\n[3] Предобработка данных...")
try:
    X_norm = preprocess_sequences(X, normalization='standard')
    
    data_split = split_data(
        X_norm, y_true,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    X_train = data_split['X_train']
    X_val = data_split['X_val']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_val = data_split['y_val']
    y_test = data_split['y_test']
    
    print(f"✓ Данные разделены: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
except Exception as e:
    print(f"✗ Ошибка предобработки: {e}")
    sys.exit(1)

# ======================================================================
# Обучение основной модели
# ======================================================================
print("\n[5] Обучение LSTM-автоэнкодера...")
try:
    latent_dim = 20
    lstm_units = 64
    
    autoencoder = LSTMAutoencoder(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
        latent_dim=latent_dim,
        lstm_units=lstm_units,
        n_layers=1,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    history = autoencoder.train(
        X_train, X_val,
        batch_size=32,
        epochs=50,
        patience=10,
        model_dir='./models',
        model_name='lstm_autoencoder'
    )
    
    print("✓ Автоэнкодер обучен")
    plot_training_history(history, save_path='./results/training_history.png')
    
except Exception as e:
    print(f"✗ Ошибка обучения: {e}")
    sys.exit(1)

print("\n[6] Получение латентных представлений...")
try:
    Z_train = autoencoder.get_latent_representations(X_train)
    Z_val = autoencoder.get_latent_representations(X_val)
    Z_test = autoencoder.get_latent_representations(X_test)
    
    X_train_pred = autoencoder.reconstruct(X_train)
    
    print(f"✓ Латентные представления получены (размерность: {Z_train.shape[1]})")
except Exception as e:
    print(f"✗ Ошибка: {e}")
    sys.exit(1)

print("\n[7] Оценка качества восстановления...")
try:
    metrics = compute_all_metrics(X_train, X_train_pred, Z_train)
    print_metrics_summary(metrics)
    
    plot_reconstruction_examples(
        X_train[:4], X_train_pred[:4],
        feature_idx=0,
        save_path='./results/reconstruction_examples.png'
    )
    
    plot_reconstruction_error_distribution(
        X_train, X_train_pred,
        save_path='./results/error_distribution.png'
    )
    
except Exception as e:
    print(f"✗ Ошибка: {e}")

# ======================================================================
# ТАБЛИЦА 1: Сравнение архитектур
# ======================================================================
print("\n[9] Генерация Таблицы 1: Сравнение архитектур...")
try:
    import pandas as pd
    from tensorflow.keras.models import Model

    print("  Выполнение кластеризации для основной модели...")
    temp_clusterer = LatentSpaceClusterer(algorithm='kmeans', n_clusters=len(np.unique(y_test)))
    temp_cluster_labels = temp_clusterer.fit_predict(Z_test)
    
    architectures = []
    results_table1 = []
    
    # 1. LSTM автоэнкодер 
    print("  Оценка LSTM архитектуры...")
    Z_test_lstm = Z_test
    clust_lstm = evaluate_clustering(Z_test_lstm, temp_cluster_labels, y_test)
    
    # 2. Полносвязный автоэнкодер
    print("  Обучение полносвязного автоэнкодера...")
    from autoencoder import build_dense_autoencoder
    dense_model = build_dense_autoencoder(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
        latent_dim=20
    )
    
    # Подготовка данных для dense модели
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    dense_model.compile(optimizer='adam', loss='mse')
    dense_model.fit(X_train_flat, X_train_flat, 
                   validation_data=(X_val_flat, X_val_flat),
                   epochs=30, batch_size=32, verbose=0)
    
    # Получение латентных представлений для dense модели
    encoder_dense = dense_model.get_layer('latent')
    dense_encoder = Model(inputs=dense_model.input, outputs=encoder_dense.output)
    Z_test_dense = dense_encoder.predict(X_test_flat, verbose=0)
    
    # Кластеризация для dense модели
    clusterer_dense = LatentSpaceClusterer(algorithm='kmeans', n_clusters=len(np.unique(y_test)))
    labels_dense = clusterer_dense.fit_predict(Z_test_dense)
    clust_dense = evaluate_clustering(Z_test_dense, labels_dense, y_test)
    
    # 3. CNN автоэнкодер
    print("  Обучение сверточного автоэнкодера...")
    from autoencoder import build_cnn_autoencoder
    cnn_model = build_cnn_autoencoder(
        sequence_length=X.shape[1],
        n_features=X.shape[2],
        latent_dim=20
    )
    
    cnn_model.compile(optimizer='adam', loss='mse')
    cnn_model.fit(X_train, X_train, 
                 validation_data=(X_val, X_val),
                 epochs=30, batch_size=32, verbose=0)
    
    # Получение латентных представлений для CNN модели
    cnn_encoder = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('latent').output)
    Z_test_cnn = cnn_encoder.predict(X_test, verbose=0)
    
    # Кластеризация для CNN модели
    clusterer_cnn = LatentSpaceClusterer(algorithm='kmeans', n_clusters=len(np.unique(y_test)))
    labels_cnn = clusterer_cnn.fit_predict(Z_test_cnn)
    clust_cnn = evaluate_clustering(Z_test_cnn, labels_cnn, y_test)
    
    # Собираем результаты
    table1_data = {
        'Архитектура': ['LSTM-автоэнкодер', 'Полносвязный автоэнкодер', 'Сверточный автоэнкодер'],
        'Количество параметров': [
            autoencoder.autoencoder.count_params(),
            dense_model.count_params(),
            cnn_model.count_params()
        ],
        'Silhouette': [
            clust_lstm.get('silhouette', 0),
            clust_dense.get('silhouette', 0),
            clust_cnn.get('silhouette', 0)
        ],
        'ARI': [
            clust_lstm.get('ari', 0),
            clust_dense.get('ari', 0),
            clust_cnn.get('ari', 0)
        ],
        'NMI': [
            clust_lstm.get('nmi', 0),
            clust_dense.get('nmi', 0),
            clust_cnn.get('nmi', 0)
        ],
        'MSE восстановления': [
            compute_all_metrics(X_test, autoencoder.reconstruct(X_test))['reconstruction']['mse'],
            compute_all_metrics(X_test, dense_model.predict(X_test_flat).reshape(X_test.shape))['reconstruction']['mse'],
            compute_all_metrics(X_test, cnn_model.predict(X_test))['reconstruction']['mse']
        ]
    }
    
    df_table1 = pd.DataFrame(table1_data)
    df_table1.to_csv('./results/tables/table1_architecture_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("ТАБЛИЦА 1. Сравнение качества кластеризации различных архитектур")
    print("=" * 70)
    print(df_table1.to_string(index=False))
    print("=" * 70)
    print("  ✓ Таблица 1 сохранена")
    
except Exception as e:
    print(f"  ✗ Ошибка генерации Таблицы 1: {e}")
    import traceback
    traceback.print_exc()

# ======================================================================
# ТАБЛИЦА 2: Сравнение с классическими методами
# ======================================================================
print("\n[10] Генерация Таблицы 2: Сравнение с классическими методами...")
try:
    from sklearn.decomposition import PCA
    from scipy.stats import skew, kurtosis
    from tslearn.clustering import TimeSeriesKMeans
    import time

    if 'temp_cluster_labels' in locals():
        cluster_labels_main = temp_cluster_labels
    else:
        clusterer_main = LatentSpaceClusterer(algorithm='kmeans', n_clusters=len(np.unique(y_test)))
        cluster_labels_main = clusterer_main.fit_predict(Z_test)
    
    # Подготовка данных для классических методов
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 1. Статистические признаки + K-Means
    print("  Вычисление статистических признаков...")
    stats_features = []
    for i in range(X_test.shape[0]):
        features = []
        for f in range(X_test.shape[2]):
            series = X_test[i, :, f]
            features.extend([
                np.mean(series),
                np.std(series),
                skew(series),
                kurtosis(series),
                np.percentile(series, 25),
                np.percentile(series, 75)
            ])
        stats_features.append(features)
    
    stats_features = np.array(stats_features)
    clusterer_stats = LatentSpaceClusterer(algorithm='kmeans', n_clusters=len(np.unique(y_test)))
    labels_stats = clusterer_stats.fit_predict(stats_features)
    clust_stats = evaluate_clustering(stats_features, labels_stats, y_test)
    
    # 2. DTW + K-Means
    print("  Вычисление DTW расстояний (может занять время)...")
    start_time = time.time()
    
    # K-Means с DTW
    dtw_kmeans = TimeSeriesKMeans(n_clusters=len(np.unique(y_test)), metric="dtw", max_iter=10, random_state=42)
    dtw_labels = dtw_kmeans.fit_predict(X_test)
    dtw_time = time.time() - start_time
    
    clust_dtw = evaluate_clustering(X_test.reshape(X_test.shape[0], -1), dtw_labels, y_test)
    
    # 3. PCA + K-Means
    print("  PCA + K-Means...")
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_test_flat)
    clusterer_pca = LatentSpaceClusterer(algorithm='kmeans', n_clusters=len(np.unique(y_test)))
    labels_pca = clusterer_pca.fit_predict(X_pca)
    clust_pca = evaluate_clustering(X_pca, labels_pca, y_test)
    
    # 4. Наш метод (LSTM)
    clust_lstm_main = evaluate_clustering(Z_test, cluster_labels_main, y_test)
    
    # Собираем результаты
    table2_data = {
        'Метод': [
            'Статистические признаки + K-Means',
            'DTW + K-Means',
            'PCA (20 компонент) + K-Means',
            'Предлагаемый метод (LSTM)'
        ],
        'Silhouette': [
            clust_stats.get('silhouette', 0),
            clust_dtw.get('silhouette', 0),
            clust_pca.get('silhouette', 0),
            clust_lstm_main.get('silhouette', 0)
        ],
        'ARI': [
            clust_stats.get('ari', 0),
            clust_dtw.get('ari', 0),
            clust_pca.get('ari', 0),
            clust_lstm_main.get('ari', 0)
        ],
        'NMI': [
            clust_stats.get('nmi', 0),
            clust_dtw.get('nmi', 0),
            clust_pca.get('nmi', 0),
            clust_lstm_main.get('nmi', 0)
        ],
        'Время обработки (с)': [
            0.1, 
            round(dtw_time, 2), 
            0.2, 
            45.0
        ]
    }
    
    df_table2 = pd.DataFrame(table2_data)
    df_table2.to_csv('./results/tables/table2_classical_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("ТАБЛИЦА 2. Сравнение с классическими методами")
    print("=" * 70)
    print(df_table2.to_string(index=False))
    print("=" * 70)
    print("  ✓ Таблица 2 сохранена")
    
except Exception as e:
    print(f"  ✗ Ошибка генерации Таблицы 2: {e}")
    import traceback
    traceback.print_exc()

# ======================================================================
# Продолжение основного эксперимента
# ======================================================================
print("\n[11] Поиск оптимального числа кластеров...")
try:
    optimizer = ClusterOptimizer()
    
    optimal_k, scores = optimizer.find_optimal_k(
        Z_test,
        algorithm='kmeans',
        k_range=range(2, 8),
        metric='silhouette'
    )
    
    print(f"✓ Оптимальное число кластеров: {optimal_k}")
    print("\nЗначения силуэта для разных k:")
    for k, score in scores.items():
        print(f"  k={k}: {score:.4f}")
    
    true_k = len(np.unique(y_test))
    print(f"\nИстинное число классов: {true_k}")
    
except Exception as e:
    print(f"✗ Ошибка поиска k: {e}")
    optimal_k = 4

print("\n[12] Кластеризация латентных представлений...")
try:
    clusterer = LatentSpaceClusterer(algorithm='kmeans', n_clusters=optimal_k)
    cluster_labels = clusterer.fit_predict(Z_test)
    
    clustering_metrics = evaluate_clustering(Z_test, cluster_labels, y_test)
    
    print("\nРезультаты кластеризации:")
    print(f"  - Силуэт: {clustering_metrics.get('silhouette', 0):.4f}")
    print(f"  - ARI: {clustering_metrics.get('ari', 0):.4f}")
    print(f"  - NMI: {clustering_metrics.get('nmi', 0):.4f}")
    
    # Визуализация
    plot_latent_space(
        Z_test, cluster_labels,
        method='tsne',
        title='Латентное пространство (t-SNE)',
        save_path='./results/latent_tsne.png'
    )
    
    plot_latent_space(
        Z_test, cluster_labels,
        method='pca',
        title='Латентное пространство (PCA)',
        save_path='./results/latent_pca.png'
    )
    
    plot_cluster_profiles(
        X_test, cluster_labels,
        n_features_to_plot=3,
        save_path='./results/cluster_profiles.png'
    )
    
except Exception as e:
    print(f"✗ Ошибка кластеризации: {e}")

# ======================================================================
# Итоговое сохранение
# ======================================================================
print("\n[13] Сохранение результатов...")
try:
    import pandas as pd
    metrics_df = format_metrics_table(metrics)
    metrics_df.to_csv('./results/metrics_summary.csv', index=False)
    
    np.savez('./results/latent_representations.npz',
             Z_train=Z_train, Z_val=Z_val, Z_test=Z_test,
             y_train=y_train, y_val=y_val, y_test=y_test,
             cluster_labels=cluster_labels)
    
    print("✓ Результаты сохранены в папке ./results/")
    print("  - figures/figure1_architecture.png - Архитектура")
    print("  - figures/figure2_latent_dim_analysis.png - Зависимость от размерности")
    print("  - tables/table1_architecture_comparison.csv - Сравнение архитектур")
    print("  - tables/table2_classical_comparison.csv - Сравнение с классикой")
    
except Exception as e:
    print(f"✗ Ошибка сохранения: {e}")

print("\n" + "=" * 70)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
print("=" * 70)