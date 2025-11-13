"""
Modelo SARIMAX (Seasonal ARIMA with eXogenous variables)
========================================================

SARIMAX é uma extensão do SARIMA que incorpora variáveis exógenas (preditoras)
no modelo. Permite modelar tanto dinâmicas internas (AR, MA) quanto efeitos
de variáveis externas.

Notação: SARIMAX(p,d,q)(P,D,Q)s + X

Componentes:
- (p,d,q): ordem não-sazonal (AR, diferenciação, MA)
- (P,D,Q)s: ordem sazonal + período
- X: variáveis exógenas (preditoras validadas)

Ideal para:
- Previsão com múltiplos preditores
- Incorporar indicadores econômicos externos
- Modelagem com variáveis de controle

Integração com validação:
- Use VariableValidator para selecionar preditores válidos
- Garante que preditores têm causalidade de Granger
- Transforma automaticamente para estacionaridade

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_model import BasePredictor
import warnings

warnings.filterwarnings('ignore')


class SARIMAXPredictor(BasePredictor):
    """
    Preditor baseado em SARIMAX (SARIMA with eXogenous variables).

    SARIMAX estende SARIMA incorporando variáveis preditoras exógenas,
    permitindo modelagem mais rica e precisa.

    Uso recomendado:
    1. Use VariableValidator para validar preditores
    2. Passe variáveis validadas como exógenas
    3. Forneça valores futuros de exógenas para previsão
    """

    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
                 exog_names: Optional[List[str]] = None,
                 trend: Optional[str] = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True,
                 name: str = "SARIMAX"):
        """
        Inicializa o modelo SARIMAX.

        Args:
            order: (p, d, q) - ordens não-sazonais
            seasonal_order: (P, D, Q, s) - ordens sazonais + período
            exog_names: Nomes das variáveis exógenas (opcional, para tracking)
            trend: Componente de tendência ('c', 't', 'ct', ou None)
            enforce_stationarity: Se True, garante estacionaridade
            enforce_invertibility: Se True, garante invertibilidade
            name: Nome do modelo
        """
        super().__init__(name)

        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_names = exog_names or []
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model = None
        self.fitted_model = None
        self.fitted_data = None
        self.fitted_exog = None

        # Validar parâmetros
        self._validate_parameters()

    def _validate_parameters(self):
        """Valida os parâmetros do modelo."""
        if len(self.order) != 3:
            raise ValueError("order deve ter 3 elementos: (p, d, q)")

        if len(self.seasonal_order) != 4:
            raise ValueError("seasonal_order deve ter 4 elementos: (P, D, Q, s)")

        p, d, q = self.order
        P, D, Q, s = self.seasonal_order

        if any(x < 0 for x in [p, d, q, P, D, Q, s]):
            raise ValueError("Todos os parâmetros devem ser não-negativos")

        if s < 2:
            raise ValueError("Período sazonal (s) deve ser >= 2")

    def fit(self,
            data: Union[np.ndarray, pd.Series],
            exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            **kwargs):
        """
        Treina o modelo SARIMAX.

        Args:
            data: Série temporal alvo (variável endógena)
            exog: Variáveis exógenas (preditoras)
                  Shape: (n_samples, n_features) ou (n_samples,) para uma variável
            **kwargs: Argumentos adicionais para o fit
        """
        try:
            # Converter para Series se necessário
            if isinstance(data, np.ndarray):
                data = pd.Series(data)

            # Processar variáveis exógenas
            if exog is not None:
                if isinstance(exog, pd.Series):
                    exog = exog.to_frame()
                elif isinstance(exog, np.ndarray):
                    if exog.ndim == 1:
                        exog = exog.reshape(-1, 1)
                    if isinstance(exog, np.ndarray):
                        exog = pd.DataFrame(exog)

                # Validar tamanhos
                if len(data) != len(exog):
                    raise ValueError(f"Tamanho incompatível: data={len(data)}, exog={len(exog)}")

                # Salvar nomes das variáveis exógenas
                if not self.exog_names:
                    self.exog_names = list(exog.columns) if hasattr(exog, 'columns') else [f'exog_{i}' for i in range(exog.shape[1])]

                self.fitted_exog = exog.copy()
            else:
                self.fitted_exog = None

            # Salvar dados de treinamento
            self.fitted_data = data.copy()

            # Criar e treinar modelo SARIMAX
            self.model = SARIMAX(
                data,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )

            # Fit
            self.fitted_model = self.model.fit()

            self.is_fitted = True

            # Armazenar informações do modelo
            self.model_info = {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'n_exog': len(self.exog_names),
                'exog_names': self.exog_names,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'n_obs': len(data),
                'converged': self.fitted_model.mle_retvals['converged'] if hasattr(self.fitted_model, 'mle_retvals') else True
            }

        except Exception as e:
            raise RuntimeError(f"Erro ao treinar SARIMAX: {str(e)}")

    def predict(self,
                steps: int = 1,
                exog: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """
        Realiza previsão para os próximos steps períodos.

        Args:
            steps: Número de períodos à frente para prever
            exog: Valores futuros das variáveis exógenas
                  OBRIGATÓRIO se o modelo foi treinado com variáveis exógenas!
                  Shape: (steps, n_features)

        Returns:
            Array com previsões
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            # Validar variáveis exógenas
            if self.fitted_exog is not None:
                if exog is None:
                    raise ValueError(
                        "Modelo foi treinado com variáveis exógenas. "
                        "Você DEVE fornecer valores futuros de exog para previsão!"
                    )

                # Processar exog
                if isinstance(exog, pd.Series):
                    exog = exog.to_frame()
                elif isinstance(exog, np.ndarray):
                    if exog.ndim == 1:
                        exog = exog.reshape(-1, 1)
                    if isinstance(exog, np.ndarray):
                        exog = pd.DataFrame(exog)

                # Validar dimensões
                expected_features = self.fitted_exog.shape[1]
                if exog.shape[1] != expected_features:
                    raise ValueError(
                        f"Número de features incompatível: esperado {expected_features}, "
                        f"recebido {exog.shape[1]}"
                    )

                if len(exog) != steps:
                    raise ValueError(
                        f"Número de observações de exog ({len(exog)}) deve ser igual "
                        f"ao número de steps ({steps})"
                    )

            # Fazer previsão
            forecast = self.fitted_model.forecast(steps=steps, exog=exog)

            # Converter para numpy array
            if isinstance(forecast, pd.Series):
                forecast = forecast.values

            return forecast

        except Exception as e:
            raise RuntimeError(f"Erro na previsão SARIMAX: {str(e)}")

    def forecast(self, data: Union[np.ndarray, pd.Series], horizon: int) -> np.ndarray:
        """
        Treina o modelo e faz previsão em uma única chamada (sem variáveis exógenas).

        NOTA: Este método não usa variáveis exógenas. Para usar exógenas,
        chame fit() e predict() separadamente.

        Args:
            data: Dados históricos para treinamento
            horizon: Horizonte de previsão (número de períodos à frente)

        Returns:
            Array com previsões
        """
        self.fit(data, exog=None)
        return self.predict(steps=horizon, exog=None)

    def predict_with_intervals(self,
                                steps: int = 1,
                                exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                                alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Realiza previsão com intervalos de confiança.

        Args:
            steps: Número de períodos à frente
            exog: Valores futuros das variáveis exógenas
            alpha: Nível de significância (default: 0.05 para 95% de confiança)

        Returns:
            Dicionário com 'mean', 'lower', 'upper'
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            # Validar e processar exog (mesma lógica do predict)
            if self.fitted_exog is not None and exog is None:
                raise ValueError("Forneça valores futuros de exog!")

            if exog is not None:
                if isinstance(exog, pd.Series):
                    exog = exog.to_frame()
                elif isinstance(exog, np.ndarray):
                    if exog.ndim == 1:
                        exog = exog.reshape(-1, 1)
                    if isinstance(exog, np.ndarray):
                        exog = pd.DataFrame(exog)

            # Obter previsão com intervalos
            forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog)

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

    def get_exog_coefficients(self) -> Dict[str, float]:
        """
        Retorna coeficientes das variáveis exógenas.

        Returns:
            Dicionário {nome_variavel: coeficiente}
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        if not self.exog_names:
            return {}

        params = self.fitted_model.params

        # Coeficientes exógenos geralmente aparecem no final
        n_exog = len(self.exog_names)
        exog_params = params[-n_exog:]

        return {name: float(coef) for name, coef in zip(self.exog_names, exog_params)}

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


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados com variáveis exógenas
    np.random.seed(42)
    n = 120  # 10 anos de dados mensais

    # Variáveis exógenas (preditoras)
    x1 = np.random.normal(0, 1, n)  # Taxa de juros
    x2 = 2 * np.sin(2 * np.pi * np.arange(n) / 12)  # Índice sazonal

    # Série temporal alvo (depende de x1 e x2)
    trend = np.linspace(0, 10, n)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / 12)
    y = np.zeros(n)

    for t in range(1, n):
        y[t] = (0.6 * y[t-1] +           # Autocorrelação
                0.3 * x1[t] +            # Efeito de x1
                0.2 * x2[t] +            # Efeito de x2
                trend[t] / 10 +
                seasonal[t] / 10 +
                np.random.normal(0, 0.5))

    # Preparar dados
    exog_data = pd.DataFrame({'x1': x1, 'x2': x2})

    # Dividir em treino e teste
    train_size = int(0.8 * len(y))
    y_train = y[:train_size]
    y_test = y[train_size:]
    exog_train = exog_data.iloc[:train_size]
    exog_test = exog_data.iloc[train_size:]

    print("EXEMPLO: Modelo SARIMAX (com variáveis exógenas)")
    print("=" * 80)
    print(f"Dados: {len(y)} observações")
    print(f"Variáveis exógenas: {list(exog_data.columns)}")
    print(f"Treino: {len(y_train)} | Teste: {len(y_test)}")
    print("=" * 80)

    # Criar e treinar modelo
    print("\n1. Treinando SARIMAX(1,1,1)(1,1,1,12) com 2 variáveis exógenas...")
    sarimax = SARIMAXPredictor(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        exog_names=['x1', 'x2']
    )
    sarimax.fit(y_train, exog=exog_train)

    # Previsão (IMPORTANTE: fornecer valores futuros de exog!)
    print("\n2. Fazendo previsão com valores futuros de exógenas...")
    forecast = sarimax.predict(steps=len(y_test), exog=exog_test)

    # Avaliar
    mape = np.mean(np.abs((y_test - forecast) / (y_test + 1e-8))) * 100
    rmse = np.sqrt(np.mean((y_test - forecast) ** 2))

    print(f"\nDesempenho:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.4f}")

    # Coeficientes das exógenas
    print("\n3. Coeficientes das variáveis exógenas:")
    coeffs = sarimax.get_exog_coefficients()
    for name, value in coeffs.items():
        print(f"  {name}: {value:.4f}")

    # Critérios de informação
    print("\n4. Critérios de Informação:")
    criteria = sarimax.get_information_criteria()
    for name, value in criteria.items():
        print(f"  {name}: {value:.2f}")

    print("\n" + "=" * 80)
    print("DICA: Use VariableValidator para selecionar preditores válidos!")
    print("=" * 80)
