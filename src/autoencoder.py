import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Dict, Optional
import os

class LSTMAutoencoder:
    """
    LSTM-автоэнкодер для сжатия многомерных временных рядов.
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        latent_dim: int = 20,
        lstm_units: int = 64,
        n_layers: int = 1,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Args:
            sequence_length: длина входной последовательности
            n_features: количество признаков
            latent_dim: размерность латентного пространства
            lstm_units: количество нейронов в LSTM-слоях
            n_layers: количество LSTM-слоев (1 или 2)
            dropout_rate: вероятность dropout
            learning_rate: скорость обучения
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        
        self._build_model()
    
    def _build_model(self):
        """Строит архитектуру автоэнкодера."""
        
        # Входной слой
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # Кодер
        if self.n_layers == 1:
            # Один LSTM-слой
            encoded = LSTM(
                self.lstm_units,
                return_sequences=False,
                dropout=self.dropout_rate
            )(input_layer)
        else:
            # Два LSTM-слоя
            x = LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate
            )(input_layer)
            encoded = LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                dropout=self.dropout_rate
            )(x)
        
        # Проекция в латентное пространство
        latent = Dense(self.latent_dim, activation='linear', name='latent')(encoded)
        
        # Декодер
        # Повторяем латентный вектор для каждого временного шага
        decoded = RepeatVector(self.sequence_length)(latent)
        
        if self.n_layers == 1:
            # Один LSTM-слой
            decoded = LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate
            )(decoded)
        else:
            # Два LSTM-слоя
            decoded = LSTM(
                self.lstm_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate
            )(decoded)
            decoded = LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate
            )(decoded)
        
        # Выходной слой (восстановление признаков)
        output_layer = TimeDistributed(Dense(self.n_features))(decoded)
        
        # Полная модель
        self.autoencoder = Model(input_layer, output_layer, name='lstm_autoencoder')
        
        # Модель кодера (для извлечения латентных представлений)
        self.encoder = Model(input_layer, latent, name='encoder')
        
        # Модель декодера (для проверки)
        decoder_input = Input(shape=(self.latent_dim,))
        decoder_sequence = RepeatVector(self.sequence_length)(decoder_input)
        
        if self.n_layers == 1:
            decoder_output = LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate
            )(decoder_sequence)
        else:
            decoder_output = LSTM(
                self.lstm_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate
            )(decoder_sequence)
            decoder_output = LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate
            )(decoder_output)
        
        decoder_output = TimeDistributed(Dense(self.n_features))(decoder_output)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        
        # Компиляция
        optimizer = Adam(learning_rate=self.learning_rate)
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Модель построена. Параметров: {self.autoencoder.count_params():,}")
        self.autoencoder.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        model_dir: str = './models',
        model_name: str = 'lstm_autoencoder'
    ) -> Dict:
        """
        Обучает автоэнкодер.
        
        Returns:
            История обучения
        """
        # Создаем директорию для моделей
        os.makedirs(model_dir, exist_ok=True)
        
        # Коллбэки
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, f'{model_name}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Обучение
        self.history = self.autoencoder.fit(
            X_train, X_train,  # Автоэнкодер обучается восстанавливать вход
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return self.history.history
    
    def get_latent_representations(self, X: np.ndarray) -> np.ndarray:
        """
        Получает латентные представления для входных данных.
        """
        if self.encoder is None:
            raise ValueError("Модель не обучена")
        
        return self.encoder.predict(X, verbose=0)
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Восстанавливает входные данные.
        """
        if self.autoencoder is None:
            raise ValueError("Модель не обучена")
        
        return self.autoencoder.predict(X, verbose=0)
    
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет ошибку восстановления для каждого объекта.
        """
        X_pred = self.reconstruct(X)
        mse = np.mean((X - X_pred) ** 2, axis=(1, 2))
        return mse
    
    def save(self, filepath: str):
        """Сохраняет модель."""
        if self.autoencoder:
            self.autoencoder.save(filepath)
            print(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str):
        """Загружает модель."""
        self.autoencoder = load_model(filepath)
        
        # Восстанавливаем энкодер
        for i, layer in enumerate(self.autoencoder.layers):
            if layer.name == 'latent':
                self.encoder = Model(
                    inputs=self.autoencoder.input,
                    outputs=layer.output
                )
                break
        
        print(f"Модель загружена из {filepath}")

# Альтернативные архитектуры для сравнения
def build_dense_autoencoder(
    sequence_length: int,
    n_features: int,
    latent_dim: int = 20
) -> Model:
    """
    Строит полносвязный автоэнкодер (без учета времени).
    """
    input_dim = sequence_length * n_features
    
    input_layer = Input(shape=(input_dim,))
    
    # Кодер
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    latent = Dense(latent_dim, activation='linear', name='latent')(encoded)
    
    # Декодер
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    model = Model(input_layer, decoded, name='dense_autoencoder')
    return model

def build_cnn_autoencoder(
    sequence_length: int,
    n_features: int,
    latent_dim: int = 20
) -> Model:
    """
    Строит сверточный автоэнкодер.
    """
    input_layer = Input(shape=(sequence_length, n_features))
    
    # Кодер
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    latent = Dense(latent_dim, activation='linear', name='latent')(x)
    
    # Декодер
    x = Dense(sequence_length // 4 * 32, activation='relu')(latent)
    x = tf.keras.layers.Reshape((sequence_length // 4, 32))(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    output = tf.keras.layers.Conv1D(n_features, 3, padding='same')(x)
    
    model = Model(input_layer, output, name='cnn_autoencoder')
    return model