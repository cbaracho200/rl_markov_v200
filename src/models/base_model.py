"""
Classe base para modelos de previsão de séries temporais.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Tuple


class BasePredictor(ABC):
    """
    Classe abstrata base para todos os modelos de previsão.

    Define a interface comum que todos os modelos devem implementar.
    """

    def __init__(self, name: str):
        """
        Inicializa o preditor.

        Args:
            name: Nome do modelo
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo com os dados fornecidos.

        Args:
            data: Dados de treinamento
            **kwargs: Argumentos adicionais específicos do modelo
        """
        pass

    @abstractmethod
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Faz previsões para os próximos passos.

        Args:
            steps: Número de passos à frente para prever

        Returns:
            Array com as previsões
        """
        pass

    @abstractmethod
    def forecast(self, data: Union[np.ndarray, pd.Series], horizon: int) -> np.ndarray:
        """
        Treina e faz previsão em uma única chamada.

        Args:
            data: Dados históricos
            horizon: Horizonte de previsão

        Returns:
            Array com as previsões
        """
        pass

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calcula métricas de avaliação.

        Args:
            y_true: Valores reais
            y_pred: Valores previstos

        Returns:
            Dicionário com métricas
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
