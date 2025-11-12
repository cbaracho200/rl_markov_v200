"""
Modelo LightGBM para previsão de séries temporais.

LightGBM é extremamente rápido e eficiente em memória.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictor


class LightGBMPredictor(BasePredictor):
    """
    Preditor baseado em LightGBM.

    LightGBM oferece:
    - Velocidade extrema
    - Baixo uso de memória
    - Excelente para datasets grandes
    - Suporte a missing values
    """

    def __init__(
        self,
        lookback: int = 12,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        name: str = "LightGBM"
    ):
        """
        Inicializa o modelo LightGBM.

        Args:
            lookback: Número de lags para features
            n_estimators: Número de árvores
            learning_rate: Taxa de aprendizado
            num_leaves: Número de folhas
            max_depth: Profundidade máxima (-1 = sem limite)
            min_child_samples: Mínimo de samples nas folhas
            subsample: Fração de samples para cada árvore
            colsample_bytree: Fração de features para cada árvore
            reg_alpha: Regularização L1
            reg_lambda: Regularização L2
            name: Nome do modelo
        """
        super().__init__(name)
        self.lookback = lookback
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model = None
        self.data = None

    def _create_features(self, data: np.ndarray) -> tuple:
        """Cria features de lag."""
        X, y = [], []

        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo LightGBM.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais
        """
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            print("⚠️  LightGBM não instalado. Instale com: pip install lightgbm")
            self.is_fitted = False
            return

        if isinstance(data, pd.Series):
            data = data.values

        self.data = data

        try:
            # Cria features
            X, y = self._create_features(data)

            if len(X) == 0:
                self.is_fitted = False
                return

            # Cria e treina modelo
            self.model = LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=42,
                verbose=-1
            )

            self.model.fit(X, y)
            self.is_fitted = True

        except Exception as e:
            print(f"Erro ao treinar LightGBM: {e}")
            self.is_fitted = False

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

        try:
            predictions = []
            current_data = self.data[-self.lookback:].copy()

            for _ in range(steps):
                # Prepara input
                X = current_data[-self.lookback:].reshape(1, -1)

                # Prediz
                pred = self.model.predict(X)[0]
                predictions.append(pred)

                # Atualiza dados
                current_data = np.append(current_data, pred)

            return np.array(predictions)

        except Exception as e:
            print(f"Erro na previsão LightGBM: {e}")
            return np.array([self.data[-1]] * steps)

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
