"""
Modelo LSTM para previsão de séries temporais.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from .base_model import BasePredictor


class LSTMNetwork(nn.Module):
    """Rede LSTM para séries temporais."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Pega apenas a última saída
        out = self.fc(lstm_out[:, -1, :])

        return out


class LSTMPredictor(BasePredictor):
    """
    Preditor baseado em LSTM (Long Short-Term Memory).

    LSTMs são excelentes para capturar dependências de longo prazo
    em séries temporais.
    """

    def __init__(
        self,
        lookback: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = 'cpu',
        name: str = "LSTM"
    ):
        """
        Inicializa o modelo LSTM.

        Args:
            lookback: Número de passos passados para considerar
            hidden_size: Tamanho da camada oculta
            num_layers: Número de camadas LSTM
            dropout: Taxa de dropout
            learning_rate: Taxa de aprendizado
            epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch
            device: Dispositivo (cpu ou cuda)
            name: Nome do modelo
        """
        super().__init__(name)

        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.scaler = MinMaxScaler()
        self.model = None
        self.data = None

    def _create_sequences(self, data: np.ndarray, lookback: int):
        """Cria sequências para treinamento."""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo LSTM.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais
        """
        if isinstance(data, pd.Series):
            data = data.values

        # Reshape se necessário
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Normaliza dados
        scaled_data = self.scaler.fit_transform(data)

        # Cria sequências
        X, y = self._create_sequences(scaled_data.flatten(), self.lookback)

        if len(X) == 0:
            print("Dados insuficientes para criar sequências")
            self.is_fitted = False
            return

        # Reshape para LSTM [batch, sequence, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Converte para tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Cria modelo
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout
        ).to(self.device)

        # Otimizador e função de perda
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Treinamento
        self.model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.arange(len(X_tensor))
            np.random.shuffle(indices)

            epoch_loss = 0
            n_batches = 0

            for start in range(0, len(X_tensor), self.batch_size):
                end = min(start + self.batch_size, len(X_tensor))
                batch_indices = indices[start:end]

                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                # Forward pass
                predictions = self.model(batch_X)

                # Loss
                loss = criterion(predictions.squeeze(), batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Época {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self.data = scaled_data.flatten()
        self.is_fitted = True

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Faz previsões para os próximos passos.

        Args:
            steps: Número de passos à frente

        Returns:
            Array com previsões
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")

        self.model.eval()

        # Usa últimos lookback pontos
        last_sequence = self.data[-self.lookback:]
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                # Prepara input
                x = torch.FloatTensor(last_sequence).reshape(1, self.lookback, 1).to(self.device)

                # Previsão
                pred = self.model(x).cpu().numpy()[0, 0]
                predictions.append(pred)

                # Atualiza sequência
                last_sequence = np.append(last_sequence[1:], pred)

        # Desnormaliza
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()

        return predictions

    def forecast(self, data: Union[np.ndarray, pd.Series], horizon: int) -> np.ndarray:
        """
        Treina e faz previsão.

        Args:
            data: Dados históricos
            horizon: Passos à frente

        Returns:
            Previsões
        """
        self.fit(data)
        return self.predict(steps=horizon)
