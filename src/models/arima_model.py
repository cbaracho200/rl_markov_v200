"""
Modelo ARIMA para previsão de séries temporais.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictor


class ARIMAPredictor(BasePredictor):
    """
    Preditor baseado em modelo ARIMA/SARIMA.

    ARIMA é um dos modelos clássicos mais eficazes para séries temporais,
    capturando tendências e sazonalidades.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 2),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        name: str = "ARIMA"
    ):
        """
        Inicializa o modelo ARIMA.

        Args:
            order: Ordem (p, d, q) do modelo ARIMA
            seasonal_order: Ordem sazonal (P, D, Q, s) para SARIMA
            name: Nome do modelo
        """
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.data = None

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo ARIMA.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais para o ARIMA
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.data = data

        try:
            if self.seasonal_order is not None:
                self.model = SARIMAX(
                    data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    **kwargs
                )
            else:
                self.model = ARIMA(
                    data,
                    order=self.order,
                    **kwargs
                )

            self.fitted_model = self.model.fit()
            self.is_fitted = True

        except Exception as e:
            print(f"Erro ao treinar ARIMA: {e}")
            # Tenta com ordem mais simples
            try:
                self.order = (1, 1, 1)
                self.model = ARIMA(data, order=self.order)
                self.fitted_model = self.model.fit()
                self.is_fitted = True
            except:
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
            forecast = self.fitted_model.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            print(f"Erro na previsão ARIMA: {e}")
            # Retorna última observação como fallback
            return np.array([self.data.iloc[-1]] * steps)

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

    def get_best_order(self, data: Union[np.ndarray, pd.Series], max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Encontra a melhor ordem usando AIC.

        Args:
            data: Dados para teste
            max_p: Máximo p
            max_q: Máximo q

        Returns:
            Melhor ordem (p, d, q)
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        best_aic = np.inf
        best_order = (1, 1, 1)

        for p in range(max_p + 1):
            for d in range(2):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue

        return best_order
