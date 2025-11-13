"""
Modelo SARIMA (Seasonal ARIMA)
==============================

SARIMA (Seasonal AutoRegressive Integrated Moving Average) é uma extensão do ARIMA
que captura padrões sazonais além da dinâmica não-sazonal.

Notação: SARIMA(p,d,q)(P,D,Q)s

Parâmetros não-sazonais:
- p: ordem autoregressiva
- d: ordem de diferenciação
- q: ordem de média móvel

Parâmetros sazonais:
- P: ordem autoregressiva sazonal
- D: ordem de diferenciação sazonal
- Q: ordem de média móvel sazonal
- s: período da sazonalidade (ex: 12 para dados mensais)

Ideal para:
- Dados com padrões sazonais claros
- Séries econômicas (vendas mensais, PIB trimestral, etc.)
- Qualquer série com ciclos regulares

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BasePredictor
import warnings

warnings.filterwarnings('ignore')


class SARIMAPredictor(BasePredictor):
    """
    Preditor baseado em SARIMA (Seasonal ARIMA).

    SARIMA modela tanto componentes não-sazonais quanto sazonais,
    sendo ideal para séries temporais com padrões cíclicos regulares.

    Exemplos de uso:
    - Dados mensais com sazonalidade anual: s=12
    - Dados trimestrais com sazonalidade anual: s=4
    - Dados semanais com sazonalidade anual: s=52
    """

    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 trend: Optional[str] = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True,
                 name: str = "SARIMA"):
        """
        Inicializa o modelo SARIMA.

        Args:
            order: (p, d, q) - ordens não-sazonais
                p: ordem AR
                d: ordem de diferenciação
                q: ordem MA
            seasonal_order: (P, D, Q, s) - ordens sazonais + período
                P: ordem AR sazonal
                D: ordem de diferenciação sazonal
                Q: ordem MA sazonal
                s: período da sazonalidade
            trend: Componente de tendência:
                - None: sem tendência
                - 'c': constante
                - 't': linear
                - 'ct': constante + linear
            enforce_stationarity: Se True, garante estacionaridade
            enforce_invertibility: Se True, garante invertibilidade
            name: Nome do modelo
        """
        super().__init__(name)

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model = None
        self.fitted_model = None
        self.fitted_data = None

        # Validar parâmetros
        self._validate_parameters()

    def _validate_parameters(self):
        """Valida os parâmetros do modelo."""
        # Validar order
        if len(self.order) != 3:
            raise ValueError("order deve ter 3 elementos: (p, d, q)")

        p, d, q = self.order
        if any(x < 0 for x in [p, d, q]):
            raise ValueError("Valores de order devem ser não-negativos")

        # Validar seasonal_order
        if len(self.seasonal_order) != 4:
            raise ValueError("seasonal_order deve ter 4 elementos: (P, D, Q, s)")

        P, D, Q, s = self.seasonal_order
        if any(x < 0 for x in [P, D, Q, s]):
            raise ValueError("Valores de seasonal_order devem ser não-negativos")

        if s < 2:
            raise ValueError("Período sazonal (s) deve ser >= 2")

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina o modelo SARIMA.

        Args:
            data: Série temporal de treinamento
            **kwargs: Argumentos adicionais para o fit (ex: disp=False)
        """
        try:
            # Converter para Series se necessário
            if isinstance(data, np.ndarray):
                data = pd.Series(data)

            # Salvar dados de treinamento
            self.fitted_data = data.copy()

            # Criar e treinar modelo SARIMA
            self.model = SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )

            # Fit (sem usar disp que foi deprecado)
            self.fitted_model = self.model.fit()

            self.is_fitted = True

            # Armazenar informações do modelo
            self.model_info = {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'n_obs': len(data),
                'converged': self.fitted_model.mle_retvals['converged'] if hasattr(self.fitted_model, 'mle_retvals') else True
            }

        except Exception as e:
            raise RuntimeError(f"Erro ao treinar SARIMA: {str(e)}")

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Realiza previsão para os próximos steps períodos.

        Args:
            steps: Número de períodos à frente para prever

        Returns:
            Array com previsões
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            # Fazer previsão
            forecast = self.fitted_model.forecast(steps=steps)

            # Converter para numpy array
            if isinstance(forecast, pd.Series):
                forecast = forecast.values

            return forecast

        except Exception as e:
            raise RuntimeError(f"Erro na previsão SARIMA: {str(e)}")

    def forecast(self, data: Union[np.ndarray, pd.Series], horizon: int) -> np.ndarray:
        """
        Treina o modelo e faz previsão em uma única chamada.

        Args:
            data: Dados históricos para treinamento
            horizon: Horizonte de previsão (número de períodos à frente)

        Returns:
            Array com previsões
        """
        self.fit(data)
        return self.predict(steps=horizon)

    def predict_with_intervals(self,
                                steps: int = 1,
                                alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Realiza previsão com intervalos de confiança.

        Args:
            steps: Número de períodos à frente
            alpha: Nível de significância (default: 0.05 para 95% de confiança)

        Returns:
            Dicionário com 'mean', 'lower', 'upper'
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            # Obter previsão com intervalos
            forecast_result = self.fitted_model.get_forecast(steps=steps)

            forecast_mean = forecast_result.predicted_mean.values
            forecast_ci = forecast_result.conf_int(alpha=alpha)

            return {
                'mean': forecast_mean,
                'lower': forecast_ci.iloc[:, 0].values,
                'upper': forecast_ci.iloc[:, 1].values,
                'confidence_level': 1 - alpha
            }

        except Exception as e:
            raise RuntimeError(f"Erro ao calcular intervalos: {str(e)}")

    def get_fitted_values(self) -> np.ndarray:
        """
        Retorna valores ajustados (fitted values) para os dados de treinamento.

        Returns:
            Array com valores ajustados
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        return self.fitted_model.fittedvalues.values

    def get_residuals(self) -> np.ndarray:
        """
        Retorna resíduos do modelo.

        Returns:
            Array com resíduos
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        return self.fitted_model.resid.values

    def get_model_summary(self) -> str:
        """
        Retorna sumário estatístico completo do modelo.

        Returns:
            String com sumário detalhado
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        return str(self.fitted_model.summary())

    def get_information_criteria(self) -> Dict[str, float]:
        """
        Retorna critérios de informação do modelo.

        Returns:
            Dicionário com AIC, BIC, HQIC
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        return {
            'AIC': self.fitted_model.aic,
            'BIC': self.fitted_model.bic,
            'HQIC': self.fitted_model.hqic,
            'Log-Likelihood': self.fitted_model.llf
        }

    def diagnose(self):
        """
        Plota diagnósticos do modelo (resíduos, ACF, etc.).

        Requer matplotlib.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            import matplotlib.pyplot as plt
            self.fitted_model.plot_diagnostics(figsize=(14, 8))
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Para diagnósticos visuais, instale: pip install matplotlib")

    @staticmethod
    def auto_select_order(data: Union[np.ndarray, pd.Series],
                          seasonal_period: int = 12,
                          max_p: int = 3,
                          max_d: int = 2,
                          max_q: int = 3,
                          max_P: int = 2,
                          max_D: int = 1,
                          max_Q: int = 2,
                          information_criterion: str = 'aic') -> Tuple[Tuple, Tuple]:
        """
        Seleciona automaticamente as ordens ótimas usando grid search.

        Args:
            data: Série temporal
            seasonal_period: Período sazonal (s)
            max_p, max_d, max_q: Máximos para ordem não-sazonal
            max_P, max_D, max_Q: Máximos para ordem sazonal
            information_criterion: 'aic', 'bic' ou 'hqic'

        Returns:
            (melhor_order, melhor_seasonal_order)
        """
        from itertools import product

        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        best_score = np.inf
        best_order = None
        best_seasonal_order = None

        # Grid search
        total_combinations = ((max_p + 1) * (max_d + 1) * (max_q + 1) *
                              (max_P + 1) * (max_D + 1) * (max_Q + 1))

        print(f"Testando {total_combinations} combinações...")

        tested = 0
        for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
            for P, D, Q in product(range(max_P + 1), range(max_D + 1), range(max_Q + 1)):
                # Pular casos triviais
                if p == d == q == P == D == Q == 0:
                    continue

                try:
                    tested += 1
                    if tested % 10 == 0:
                        print(f"  Progresso: {tested}/{total_combinations}", end='\r')

                    model = SARIMAX(
                        data,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fitted = model.fit(disp=False)

                    # Obter critério de informação
                    if information_criterion == 'aic':
                        score = fitted.aic
                    elif information_criterion == 'bic':
                        score = fitted.bic
                    elif information_criterion == 'hqic':
                        score = fitted.hqic
                    else:
                        score = fitted.aic

                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                        best_seasonal_order = (P, D, Q, seasonal_period)

                except:
                    continue

        print(f"\nMelhor modelo encontrado:")
        print(f"  Order: {best_order}")
        print(f"  Seasonal Order: {best_seasonal_order}")
        print(f"  {information_criterion.upper()}: {best_score:.2f}")

        return best_order, best_seasonal_order


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados com sazonalidade
    np.random.seed(42)
    n = 120  # 10 anos de dados mensais

    # Componentes
    trend = np.linspace(0, 10, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)  # Sazonalidade anual
    noise = np.random.normal(0, 1, n)

    # Série temporal
    data = trend + seasonal + noise

    # Dividir em treino e teste
    train_size = int(0.8 * len(data))
    train = data[:train_size]
    test = data[train_size:]

    print("EXEMPLO: Modelo SARIMA")
    print("=" * 80)
    print(f"Dados: {len(data)} observações (sazonalidade anual)")
    print(f"Treino: {len(train)} | Teste: {len(test)}")
    print("=" * 80)

    # Criar e treinar modelo
    print("\n1. Treinando SARIMA(1,1,1)(1,1,1,12)...")
    sarima = SARIMAPredictor(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    sarima.fit(train)

    # Previsão
    print("\n2. Fazendo previsão...")
    forecast = sarima.predict(steps=len(test))

    # Previsão com intervalos
    print("\n3. Previsão com intervalos de confiança...")
    forecast_intervals = sarima.predict_with_intervals(steps=len(test), alpha=0.05)

    # Avaliar
    mape = np.mean(np.abs((test - forecast) / (test + 1e-8))) * 100
    rmse = np.sqrt(np.mean((test - forecast) ** 2))

    print(f"\nDesempenho:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.4f}")

    # Critérios de informação
    print("\n4. Critérios de Informação:")
    criteria = sarima.get_information_criteria()
    for name, value in criteria.items():
        print(f"  {name}: {value:.2f}")

    # Seleção automática (comentado pois leva tempo)
    # print("\n5. Seleção automática de ordem (pode demorar)...")
    # best_order, best_seasonal = SARIMAPredictor.auto_select_order(
    #     train,
    #     seasonal_period=12,
    #     max_p=2, max_d=1, max_q=2,
    #     max_P=1, max_D=1, max_Q=1
    # )
