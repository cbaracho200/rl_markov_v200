"""
Modelo Prophet (Facebook) para previsão de séries temporais.

Prophet é robusto a dados faltantes e mudanças de tendência.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictor


class ProphetPredictor(BasePredictor):
    """
    Preditor baseado em Prophet do Facebook.

    Prophet é especialmente bom para:
    - Séries com forte sazonalidade
    - Múltiplas sazonalidades (diária, semanal, anual)
    - Feriados e eventos especiais
    - Dados faltantes
    - Mudanças de tendência
    """

    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        yearly_seasonality: Union[bool, str] = 'auto',
        weekly_seasonality: Union[bool, str] = 'auto',
        daily_seasonality: Union[bool, str] = 'auto',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        name: str = "Prophet"
    ):
        """
        Inicializa o modelo Prophet.

        Args:
            seasonality_mode: 'additive' ou 'multiplicative'
            yearly_seasonality: Sazonalidade anual
            weekly_seasonality: Sazonalidade semanal
            daily_seasonality: Sazonalidade diária
            changepoint_prior_scale: Flexibilidade da tendência
            seasonality_prior_scale: Força da sazonalidade
            name: Nome do modelo
        """
        super().__init__(name)
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
        self.data = None

    def fit(self, data: Union[np.ndarray, pd.Series, pd.DataFrame], **kwargs):
        """
        Treina o modelo Prophet.

        Args:
            data: Série temporal para treinamento
            **kwargs: Argumentos adicionais
        """
        try:
            from prophet import Prophet
        except ImportError:
            print("⚠️  Prophet não instalado. Instale com: pip install prophet")
            self.is_fitted = False
            return

        # Prepara dados no formato do Prophet
        if isinstance(data, np.ndarray):
            df = pd.DataFrame({
                'ds': pd.date_range(start='2000-01-01', periods=len(data), freq='M'),
                'y': data
            })
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({
                'ds': pd.date_range(start='2000-01-01', periods=len(data), freq='M'),
                'y': data.values
            })
        elif isinstance(data, pd.DataFrame):
            if 'ds' not in data.columns or 'y' not in data.columns:
                # Assume primeira coluna é data, segunda é valor
                df = data.copy()
                df.columns = ['ds', 'y']
            else:
                df = data.copy()
        else:
            raise ValueError("Data deve ser np.ndarray, pd.Series ou pd.DataFrame")

        self.data = df

        try:
            # Cria modelo
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale
            )

            # Treina
            self.model.fit(df)
            self.is_fitted = True

        except Exception as e:
            print(f"Erro ao treinar Prophet: {e}")
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
            # Cria datas futuras
            future = self.model.make_future_dataframe(periods=steps, freq='M')

            # Faz previsão
            forecast = self.model.predict(future)

            # Retorna apenas os valores futuros
            predictions = forecast['yhat'].values[-steps:]

            return np.array(predictions)

        except Exception as e:
            print(f"Erro na previsão Prophet: {e}")
            # Retorna última observação como fallback
            return np.array([self.data['y'].iloc[-1]] * steps)

    def forecast(self, data: Union[np.ndarray, pd.Series, pd.DataFrame], horizon: int) -> np.ndarray:
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
