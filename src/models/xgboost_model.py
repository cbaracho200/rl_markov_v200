"""
Modelo XGBoost para previsão de séries temporais.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .base_model import BasePredictor


class XGBoostPredictor(BasePredictor):
    """
    Preditor baseado em XGBoost.

    XGBoost é um modelo de gradient boosting extremamente eficaz,
    especialmente para capturar padrões não-lineares em dados.
    """

    def __init__(
        self,
        lookback: int = 24,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        name: str = "XGBoost"
    ):
        """
        Inicializa o modelo XGBoost.

        Args:
            lookback: Número de passos passados para usar como features
            n_estimators: Número de árvores
            max_depth: Profundidade máxima das árvores
            learning_rate: Taxa de aprendizado
            subsample: Fração de amostras para cada árvore
            colsample_bytree: Fração de features para cada árvore
            name: Nome do modelo
        """
        super().__init__(name)

        self.lookback = lookback
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.model = None
        self.scaler = StandardScaler()
        self.data = None

    def _create_features(self, data: np.ndarray, lookback: int):
        """
        Cria features de lag para XGBoost.

        Args:
            data: Série temporal
            lookback: Número de lags

        Returns:
            X: Features
            y: Targets
        """
        X, y = [], []

        for i in range(lookback, len(data)):
            # Features: lags + features estatísticas
            lags = data[i - lookback:i]

            features = list(lags)

            # Adiciona features estatísticas
            features.extend([
                np.mean(lags),
                np.std(lags),
                np.max(lags),
                np.min(lags),
                lags[-1] - lags[0],  # Tendência
                np.mean(np.diff(lags))  # Velocidade
            ])

            X.append(features)
            y.append(data[i])

        return np.array(X), np.array(y)

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo XGBoost.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais para XGBoost
        """
        if isinstance(data, pd.Series):
            data = data.values

        if len(data.shape) > 1:
            data = data.flatten()

        # Normaliza dados
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

        # Cria features
        X, y = self._create_features(data_scaled, self.lookback)

        if len(X) == 0:
            print("Dados insuficientes para criar features")
            self.is_fitted = False
            return

        # Cria e treina modelo
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            **kwargs
        )

        self.model.fit(X, y, verbose=False)

        self.data = data_scaled
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

        predictions = []
        last_sequence = self.data[-self.lookback:].copy()

        for _ in range(steps):
            # Cria features
            lags = last_sequence[-self.lookback:]

            features = list(lags)
            features.extend([
                np.mean(lags),
                np.std(lags),
                np.max(lags),
                np.min(lags),
                lags[-1] - lags[0],
                np.mean(np.diff(lags))
            ])

            # Previsão
            X = np.array(features).reshape(1, -1)
            pred = self.model.predict(X)[0]

            predictions.append(pred)

            # Atualiza sequência
            last_sequence = np.append(last_sequence, pred)

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

    def feature_importance(self) -> pd.DataFrame:
        """
        Retorna importância das features.

        Returns:
            DataFrame com importância
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado.")

        importance = self.model.feature_importances_

        # Nomes das features
        feature_names = [f'lag_{i+1}' for i in range(self.lookback)]
        feature_names.extend(['mean', 'std', 'max', 'min', 'trend', 'velocity'])

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df
