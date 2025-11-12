"""
Modelo CatBoost para previsão de séries temporais.

CatBoost é um algoritmo de gradient boosting state-of-the-art.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictor


class CatBoostPredictor(BasePredictor):
    """
    Preditor baseado em CatBoost.

    CatBoost oferece:
    - Melhor performance que XGBoost/LightGBM em muitos casos
    - Suporte nativo a features categóricas
    - Menor overfitting
    - Treinamento mais rápido
    """

    def __init__(
        self,
        lookback: int = 12,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        name: str = "CatBoost"
    ):
        """
        Inicializa o modelo CatBoost.

        Args:
            lookback: Número de lags para features
            iterations: Número de árvores
            learning_rate: Taxa de aprendizado
            depth: Profundidade das árvores
            l2_leaf_reg: Regularização L2 nas folhas
            random_strength: Força da aleatoriedade
            bagging_temperature: Temperatura do bagging
            name: Nome do modelo
        """
        super().__init__(name)
        self.lookback = lookback
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
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
        Treina o modelo CatBoost.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais
        """
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            print("⚠️  CatBoost não instalado. Instale com: pip install catboost")
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
            self.model = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_strength=self.random_strength,
                bagging_temperature=self.bagging_temperature,
                verbose=False,
                random_seed=42
            )

            self.model.fit(X, y)
            self.is_fitted = True

        except Exception as e:
            print(f"Erro ao treinar CatBoost: {e}")
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
            print(f"Erro na previsão CatBoost: {e}")
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
