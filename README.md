# LSTM-Autoencoder for multivariate time series clustering

Unsupervised clustering of multivariate time series using LSTM-autoencoder with latent space compression and K-Means clustering.

## Features

- LSTM-based autoencoder with configurable architecture (1-3 layers, adjustable latent dimensions)
- Automatic feature extraction from temporal data (no manual engineering)
- Comprehensive evaluation metrics (ARI, NMI, silhouette, MSE, MAE, R²)
- Visualization tools (latent space PCA/t-SNE, cluster profiles, error distributions)
- Comparison with baseline methods (DTW, PCA, statistical features)

## Pipeline

1. Generate/load multivariate time series data
2. Train LSTM-autoencoder to compress sequences into latent vectors
3. Extract latent representations
4. Find optimal number of clusters using silhouette score
5. Apply K-Means clustering in latent space
6. Evaluate clustering quality (ARI, NMI, silhouette)

## Results

| Architecture    | Silhouette | ARI | MSE   |
|-----------------|------------|-----|-------|
| LSTM (improved) | 0.770      | 1.0 | 0.581 |
| CNN             | 0.729      | 1.0 | 0.692 |
| Dense           | 0.675      | 1.0 | 0.359 |
