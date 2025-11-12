"""
Modelo VAR (Vector Autoregression)
==================================

VAR é um modelo multivariado que modela simultaneamente múltiplas séries temporais
inter-relacionadas. Cada variável é modelada como uma função linear de seus próprios
valores passados e dos valores passados de todas as outras variáveis.

Notação: VAR(p)
- p: número de lags

Características:
- Modela relações dinâmicas entre múltiplas variáveis
- Captura interdependências e feedback loops
- Ideal para sistemas econômicos com variáveis interdependentes
- Requer todas as variáveis estacionárias

Ideal para:
- Análise de sistemas econômicos (PIB, inflação, taxa de juros)
- Previsão de múltiplas séries relacionadas simultaneamente
- Análise de impulso-resposta
- Decomposição de variância

Integração com validação:
- Use GrangerCausality para identificar relações entre variáveis
- Use StationarityTests para garantir estacionaridade
- Todas as variáveis devem ser estacionárias!

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, List, Tuple
from statsmodels.tsa.api import VAR as VAR_Model
from statsmodels.tsa.vector_ar.var_model import VARResults
from ..base_predictor import BasePredictor
import warnings

warnings.filterwarnings('ignore')


class VARPredictor(BasePredictor):
    """
    Preditor baseado em VAR (Vector Autoregression).

    VAR modela múltiplas séries temporais simultaneamente, capturando
    interdependências e permitindo previsão conjunta.

    IMPORTANTE:
    - Todas as variáveis DEVEM ser estacionárias
    - Use VariableValidator para transformar dados se necessário
    """

    def __init__(self,
                 maxlags: Optional[int] = None,
                 ic: str = 'aic',
                 trend: str = 'c',
                 **kwargs):
        """
        Inicializa o modelo VAR.

        Args:
            maxlags: Número máximo de lags a considerar (None = seleção automática)
            ic: Critério de informação para seleção de lags:
                - 'aic': Akaike Information Criterion
                - 'bic': Bayesian Information Criterion
                - 'hqic': Hannan-Quinn Information Criterion
                - 'fpe': Final Prediction Error
            trend: Componente de tendência:
                - 'c': constante (padrão)
                - 'ct': constante + tendência linear
                - 'ctt': constante + tendência linear + quadrática
                - 'n': sem termo de tendência
        """
        super().__init__(**kwargs)

        self.maxlags = maxlags
        self.ic = ic
        self.trend = trend

        self.model = None
        self.fitted_model: Optional[VARResults] = None
        self.fitted_data = None
        self.variable_names = []
        self.selected_lag = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs):
        """
        Treina o modelo VAR.

        Args:
            data: DataFrame ou array com múltiplas séries temporais
                  Shape: (n_samples, n_variables)
                  Colunas devem ser as diferentes variáveis
            **kwargs: Argumentos adicionais para o fit
        """
        try:
            # Converter para DataFrame se necessário
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    raise ValueError("VAR requer múltiplas variáveis. Use ARIMA para univariada.")

                n_vars = data.shape[1]
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(n_vars)])

            if data.shape[1] < 2:
                raise ValueError("VAR requer pelo menos 2 variáveis.")

            # Salvar nomes das variáveis
            self.variable_names = list(data.columns)

            # Salvar dados de treinamento
            self.fitted_data = data.copy()

            # Criar modelo VAR
            self.model = VAR_Model(data)

            # Selecionar ordem de lag
            if self.maxlags is None:
                # Seleção automática
                lag_order_results = self.model.select_order(maxlags=15)
                self.selected_lag = getattr(lag_order_results, self.ic)

                print(f"Ordem de lag selecionada automaticamente: {self.selected_lag} (critério: {self.ic.upper()})")
            else:
                self.selected_lag = self.maxlags

            # Treinar modelo
            self.fitted_model = self.model.fit(maxlags=self.selected_lag, trend=self.trend)

            self.is_trained = True

            # Armazenar informações do modelo
            self.model_info = {
                'n_variables': len(self.variable_names),
                'variable_names': self.variable_names,
                'lag_order': self.selected_lag,
                'trend': self.trend,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'fpe': self.fitted_model.fpe,
                'n_obs': self.fitted_model.nobs
            }

        except Exception as e:
            raise RuntimeError(f"Erro ao treinar VAR: {str(e)}")

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Realiza previsão para os próximos steps períodos.

        Args:
            steps: Número de períodos à frente para prever

        Returns:
            Array com previsões para todas as variáveis
            Shape: (steps, n_variables)
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            # Obter últimas observações (necessário para VAR)
            lag_order = self.fitted_model.k_ar
            last_obs = self.fitted_data.values[-lag_order:]

            # Fazer previsão
            forecast = self.fitted_model.forecast(y=last_obs, steps=steps)

            return forecast

        except Exception as e:
            raise RuntimeError(f"Erro na previsão VAR: {str(e)}")

    def predict_single_variable(self,
                                 variable_name: str,
                                 steps: int = 1) -> np.ndarray:
        """
        Realiza previsão para uma única variável específica.

        Args:
            variable_name: Nome da variável a prever
            steps: Número de períodos à frente

        Returns:
            Array com previsões para a variável especificada
            Shape: (steps,)
        """
        if variable_name not in self.variable_names:
            raise ValueError(f"Variável '{variable_name}' não encontrada. "
                           f"Variáveis disponíveis: {self.variable_names}")

        # Prever todas as variáveis
        forecast_all = self.predict(steps=steps)

        # Extrair variável específica
        var_index = self.variable_names.index(variable_name)
        return forecast_all[:, var_index]

    def get_granger_causality_matrix(self, maxlag: Optional[int] = None) -> pd.DataFrame:
        """
        Calcula matriz de causalidade de Granger entre todas as variáveis.

        Args:
            maxlag: Número máximo de lags para teste (se None, usa lag do modelo)

        Returns:
            DataFrame matriz onde [i,j] = p-value do teste "i causa j"
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        if maxlag is None:
            maxlag = self.selected_lag

        try:
            # Testar causalidade para cada par de variáveis
            n_vars = len(self.variable_names)
            causality_matrix = np.ones((n_vars, n_vars))  # Diagonal = 1

            for i, caused_var in enumerate(self.variable_names):
                for j, causing_var in enumerate(self.variable_names):
                    if i != j:
                        # Testar se causing_var causa caused_var
                        test_result = self.fitted_model.test_causality(
                            caused=caused_var,
                            causing=causing_var,
                            kind='f'
                        )
                        causality_matrix[j, i] = test_result.pvalue

            # Criar DataFrame
            df = pd.DataFrame(
                causality_matrix,
                index=self.variable_names,
                columns=self.variable_names
            )

            return df

        except Exception as e:
            print(f"Aviso: Não foi possível calcular matriz de causalidade: {str(e)}")
            return pd.DataFrame()

    def get_impulse_response(self,
                             periods: int = 10,
                             impulse: Optional[str] = None,
                             response: Optional[str] = None) -> np.ndarray:
        """
        Calcula funções de impulso-resposta (IRF).

        IRF mostra como um choque em uma variável afeta outras variáveis ao longo do tempo.

        Args:
            periods: Número de períodos para calcular a resposta
            impulse: Variável que recebe o impulso (None = todas)
            response: Variável cuja resposta é medida (None = todas)

        Returns:
            Array com IRF
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            irf = self.fitted_model.irf(periods=periods)

            # Se especificado, extrair apenas impulso/resposta específicos
            if impulse and response:
                imp_idx = self.variable_names.index(impulse)
                resp_idx = self.variable_names.index(response)
                return irf.irfs[:, resp_idx, imp_idx]

            return irf.irfs

        except Exception as e:
            raise RuntimeError(f"Erro ao calcular IRF: {str(e)}")

    def plot_impulse_response(self,
                              impulse: Optional[str] = None,
                              response: Optional[str] = None,
                              periods: int = 10):
        """
        Plota funções de impulso-resposta.

        Args:
            impulse: Variável que recebe o impulso (None = todas)
            response: Variável cuja resposta é medida (None = todas)
            periods: Número de períodos
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            irf = self.fitted_model.irf(periods=periods)

            if impulse and response:
                # Plot específico
                irf.plot(impulse=impulse, response=response)
            else:
                # Plot de todas as combinações
                irf.plot()

        except ImportError:
            print("Para plotar IRF, instale: pip install matplotlib")
        except Exception as e:
            print(f"Erro ao plotar IRF: {str(e)}")

    def get_forecast_error_variance_decomposition(self,
                                                   periods: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Calcula decomposição de variância do erro de previsão (FEVD).

        FEVD mostra a proporção da variância do erro de previsão de cada variável
        que é explicada por choques em cada variável do sistema.

        Args:
            periods: Número de períodos para decomposição

        Returns:
            Dicionário {variável: DataFrame com decomposição}
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        try:
            fevd = self.fitted_model.fevd(periods=periods)

            result = {}
            for i, var_name in enumerate(self.variable_names):
                # Criar DataFrame para cada variável
                decomp = pd.DataFrame(
                    fevd.decomp[:, i, :],
                    columns=self.variable_names
                )
                result[var_name] = decomp

            return result

        except Exception as e:
            raise RuntimeError(f"Erro ao calcular FEVD: {str(e)}")

    def get_model_summary(self) -> str:
        """
        Retorna sumário estatístico completo do modelo.

        Returns:
            String com sumário detalhado
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        return str(self.fitted_model.summary())

    def get_information_criteria(self) -> Dict[str, float]:
        """
        Retorna critérios de informação do modelo.

        Returns:
            Dicionário com AIC, BIC, HQIC, FPE
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")

        return {
            'AIC': self.fitted_model.aic,
            'BIC': self.fitted_model.bic,
            'HQIC': self.fitted_model.hqic,
            'FPE': self.fitted_model.fpe
        }


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados multivariados inter-relacionados
    np.random.seed(42)
    n = 200

    # Sistema com 3 variáveis interdependentes
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)

    for t in range(2, n):
        # y1 depende de seus próprios lags e de y2
        y1[t] = 0.5 * y1[t-1] + 0.3 * y2[t-1] + np.random.normal(0, 0.5)

        # y2 depende de seus próprios lags e de y1
        y2[t] = 0.4 * y2[t-1] + 0.2 * y1[t-1] + 0.1 * y3[t-1] + np.random.normal(0, 0.5)

        # y3 depende principalmente de si mesmo
        y3[t] = 0.6 * y3[t-1] + 0.15 * y1[t-2] + np.random.normal(0, 0.5)

    # Criar DataFrame
    data = pd.DataFrame({
        'GDP': y1,
        'Inflation': y2,
        'Interest_Rate': y3
    })

    # Dividir em treino e teste
    train_size = int(0.8 * len(data))
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    print("EXEMPLO: Modelo VAR (Vector Autoregression)")
    print("=" * 80)
    print(f"Variáveis: {list(data.columns)}")
    print(f"Observações: {len(data)}")
    print(f"Treino: {len(train)} | Teste: {len(test)}")
    print("=" * 80)

    # Criar e treinar modelo
    print("\n1. Treinando modelo VAR com seleção automática de lag...")
    var = VARPredictor(maxlags=None, ic='aic')
    var.fit(train)

    # Previsão
    print("\n2. Fazendo previsão...")
    forecast_all = var.predict(steps=len(test))

    # Previsão para variável específica
    forecast_gdp = var.predict_single_variable('GDP', steps=len(test))

    # Avaliar GDP
    actual_gdp = test['GDP'].values
    mape = np.mean(np.abs((actual_gdp - forecast_gdp) / (actual_gdp + 1e-8))) * 100
    rmse = np.sqrt(np.mean((actual_gdp - forecast_gdp) ** 2))

    print(f"\nDesempenho (GDP):")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.4f}")

    # Critérios de informação
    print("\n3. Critérios de Informação:")
    criteria = var.get_information_criteria()
    for name, value in criteria.items():
        print(f"  {name}: {value:.4f}")

    # Matriz de causalidade de Granger
    print("\n4. Matriz de Causalidade de Granger (p-values):")
    causality = var.get_granger_causality_matrix()
    print(causality.round(4))
    print("\n(Linhas causam Colunas. P-value < 0.05 = causalidade significativa)")

    # FEVD
    print("\n5. Decomposição de Variância do Erro de Previsão (FEVD):")
    fevd = var.get_forecast_error_variance_decomposition(periods=5)
    print("\nGDP no período 5:")
    print(fevd['GDP'].iloc[-1].round(4))

    print("\n" + "=" * 80)
    print("DICA: Use VariableValidator para garantir estacionaridade!")
    print("=" * 80)
