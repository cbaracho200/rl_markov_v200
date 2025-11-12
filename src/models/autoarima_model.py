"""
Modelo AutoARIMA com busca automática de hiperparâmetros.

Usa pmdarima para encontrar automaticamente os melhores parâmetros ARIMA.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictor


class AutoARIMAPredictor(BasePredictor):
    """
    Preditor baseado em AutoARIMA com busca automática de hiperparâmetros.

    AutoARIMA:
    - Encontra automaticamente os melhores parâmetros (p, d, q)
    - Suporta sazonalidade automática
    - Usa testes estatísticos (ADF, KPSS)
    - Mais robusto que ARIMA manual
    """

    def __init__(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        seasonal: bool = True,
        m: int = 12,
        stepwise: bool = True,
        information_criterion: str = 'aic',
        trace: bool = False,
        name: str = "AutoARIMA"
    ):
        """
        Inicializa o modelo AutoARIMA.

        Args:
            max_p: Máximo p (ordem AR)
            max_d: Máximo d (diferenciação)
            max_q: Máximo q (ordem MA)
            max_P: Máximo P sazonal
            max_D: Máximo D sazonal
            max_Q: Máximo Q sazonal
            seasonal: Se True, usa SARIMA
            m: Período sazonal (12 para mensal)
            stepwise: Se True, usa stepwise search (mais rápido)
            information_criterion: 'aic', 'bic', ou 'hqic'
            trace: Se True, mostra progresso
            name: Nome do modelo
        """
        super().__init__(name)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.seasonal = seasonal
        self.m = m
        self.stepwise = stepwise
        self.information_criterion = information_criterion
        self.trace = trace
        self.model = None
        self.fitted_model = None
        self.data = None
        self.best_order = None
        self.best_seasonal_order = None

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo AutoARIMA com busca automática.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais
        """
        try:
            from pmdarima import auto_arima
        except ImportError:
            print("⚠️  pmdarima não instalado. Instale com: pip install pmdarima")
            self.is_fitted = False
            return

        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.data = data

        try:
            # Busca automática de parâmetros
            self.fitted_model = auto_arima(
                data,
                start_p=0, max_p=self.max_p,
                start_q=0, max_q=self.max_q,
                max_d=self.max_d,
                start_P=0, max_P=self.max_P,
                start_Q=0, max_Q=self.max_Q,
                max_D=self.max_D,
                seasonal=self.seasonal,
                m=self.m,
                stepwise=self.stepwise,
                information_criterion=self.information_criterion,
                trace=self.trace,
                error_action='ignore',
                suppress_warnings=True,
                random_state=42
            )

            # Salva melhores parâmetros
            self.best_order = self.fitted_model.order
            self.best_seasonal_order = self.fitted_model.seasonal_order if self.seasonal else None

            self.is_fitted = True

            if self.trace:
                print(f"\n✅ Melhor modelo encontrado:")
                print(f"   Ordem: {self.best_order}")
                if self.best_seasonal_order:
                    print(f"   Ordem Sazonal: {self.best_seasonal_order}")
                print(f"   {self.information_criterion.upper()}: {self.fitted_model.aic():.2f}")

        except Exception as e:
            print(f"Erro ao treinar AutoARIMA: {e}")
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
            forecast = self.fitted_model.predict(n_periods=steps)
            return np.array(forecast)

        except Exception as e:
            print(f"Erro na previsão AutoARIMA: {e}")
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

    def get_best_parameters(self) -> dict:
        """
        Retorna os melhores parâmetros encontrados.

        Returns:
            Dicionário com parâmetros
        """
        if not self.is_fitted:
            return {}

        params = {
            'order': self.best_order,
            'seasonal_order': self.best_seasonal_order,
            'aic': self.fitted_model.aic(),
            'bic': self.fitted_model.bic()
        }

        return params

    def update(self, new_data: Union[np.ndarray, pd.Series]):
        """
        Atualiza o modelo com novos dados sem retreinar do zero.

        Args:
            new_data: Novos pontos de dados
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")

        try:
            self.fitted_model.update(new_data)
            self.data = pd.concat([self.data, pd.Series(new_data)])
        except Exception as e:
            print(f"Erro ao atualizar AutoARIMA: {e}")
