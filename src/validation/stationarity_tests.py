"""
Módulo de Testes de Estacionaridade
===================================

Este módulo implementa testes estatísticos rigorosos para verificar se uma série temporal
é estacionária, incluindo:
- Teste ADF (Augmented Dickey-Fuller)
- Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
- Teste Phillips-Perron

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, Tuple
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
import warnings

warnings.filterwarnings('ignore')


class StationarityTests:
    """
    Classe para realizar testes de estacionaridade em séries temporais.

    Testes implementados:
    1. ADF (Augmented Dickey-Fuller): H0 = série tem raiz unitária (não estacionária)
    2. KPSS: H0 = série é estacionária
    3. Phillips-Perron: H0 = série tem raiz unitária (não estacionária)

    Interpretação:
    - ADF: p-value < 0.05 → rejeitar H0 → série é estacionária
    - KPSS: p-value > 0.05 → não rejeitar H0 → série é estacionária
    - PP: p-value < 0.05 → rejeitar H0 → série é estacionária
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Inicializa o testador de estacionaridade.

        Args:
            significance_level: Nível de significância (default: 0.05 = 5%)
        """
        self.significance_level = significance_level
        self.results = {}

    def adf_test(self,
                 data: Union[np.ndarray, pd.Series],
                 max_lags: Optional[int] = None,
                 regression: str = 'c') -> Dict:
        """
        Teste Augmented Dickey-Fuller (ADF).

        H0: A série tem uma raiz unitária (não estacionária)
        H1: A série não tem raiz unitária (estacionária)

        Args:
            data: Série temporal
            max_lags: Número máximo de lags (None = automático)
            regression: Tipo de regressão:
                - 'c': constante apenas
                - 'ct': constante + tendência
                - 'ctt': constante + tendência linear + quadrática
                - 'n': sem constante ou tendência

        Returns:
            Dicionário com resultados do teste
        """
        try:
            # Converter para array se necessário
            if isinstance(data, pd.Series):
                data = data.values

            # Remover NaN
            data = data[~np.isnan(data)]

            # Executar teste ADF
            adf_result = adfuller(data, maxlag=max_lags, regression=regression, autolag='AIC')

            # Extrair resultados
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            used_lag = adf_result[2]
            n_obs = adf_result[3]
            critical_values = adf_result[4]

            # Interpretar resultado
            is_stationary = p_value < self.significance_level

            result = {
                'test_name': 'ADF (Augmented Dickey-Fuller)',
                'statistic': adf_statistic,
                'p_value': p_value,
                'used_lag': used_lag,
                'n_obs': n_obs,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': 'Estacionária' if is_stationary else 'Não Estacionária',
                'interpretation': self._interpret_adf(p_value, adf_statistic, critical_values)
            }

            return result

        except Exception as e:
            return {
                'test_name': 'ADF',
                'error': str(e),
                'is_stationary': None
            }

    def kpss_test(self,
                  data: Union[np.ndarray, pd.Series],
                  regression: str = 'c',
                  lags: Optional[int] = None) -> Dict:
        """
        Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin).

        H0: A série é estacionária
        H1: A série não é estacionária

        IMPORTANTE: Este teste tem hipótese nula OPOSTA ao ADF!

        Args:
            data: Série temporal
            regression: Tipo de regressão:
                - 'c': constante (level stationarity)
                - 'ct': constante + tendência (trend stationarity)
            lags: Número de lags (None = automático)

        Returns:
            Dicionário com resultados do teste
        """
        try:
            # Converter para array se necessário
            if isinstance(data, pd.Series):
                data = data.values

            # Remover NaN
            data = data[~np.isnan(data)]

            # Executar teste KPSS
            kpss_result = kpss(data, regression=regression, nlags=lags)

            # Extrair resultados
            kpss_statistic = kpss_result[0]
            p_value = kpss_result[1]
            used_lag = kpss_result[2]
            critical_values = kpss_result[3]

            # Interpretar resultado (H0: estacionária)
            is_stationary = p_value > self.significance_level

            result = {
                'test_name': 'KPSS (Kwiatkowski-Phillips-Schmidt-Shin)',
                'statistic': kpss_statistic,
                'p_value': p_value,
                'used_lag': used_lag,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': 'Estacionária' if is_stationary else 'Não Estacionária',
                'interpretation': self._interpret_kpss(p_value, kpss_statistic, critical_values)
            }

            return result

        except Exception as e:
            return {
                'test_name': 'KPSS',
                'error': str(e),
                'is_stationary': None
            }

    def phillips_perron_test(self,
                             data: Union[np.ndarray, pd.Series],
                             lags: Optional[int] = None,
                             trend: str = 'c') -> Dict:
        """
        Teste Phillips-Perron (PP).

        H0: A série tem uma raiz unitária (não estacionária)
        H1: A série não tem raiz unitária (estacionária)

        Semelhante ao ADF, mas mais robusto a heterocedasticidade.

        Args:
            data: Série temporal
            lags: Número de lags (None = automático)
            trend: Tipo de tendência:
                - 'c': constante apenas
                - 'ct': constante + tendência
                - 'n': sem constante ou tendência

        Returns:
            Dicionário com resultados do teste
        """
        try:
            # Converter para array se necessário
            if isinstance(data, pd.Series):
                data = data.values

            # Remover NaN
            data = data[~np.isnan(data)]

            # Executar teste Phillips-Perron
            pp_test = PhillipsPerron(data, lags=lags, trend=trend)

            # Extrair resultados
            pp_statistic = pp_test.stat
            p_value = pp_test.pvalue
            critical_values = pp_test.critical_values

            # Interpretar resultado
            is_stationary = p_value < self.significance_level

            result = {
                'test_name': 'Phillips-Perron',
                'statistic': pp_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': 'Estacionária' if is_stationary else 'Não Estacionária',
                'interpretation': self._interpret_pp(p_value, pp_statistic, critical_values)
            }

            return result

        except Exception as e:
            return {
                'test_name': 'Phillips-Perron',
                'error': str(e),
                'is_stationary': None
            }

    def run_all_tests(self,
                      data: Union[np.ndarray, pd.Series],
                      verbose: bool = True) -> Dict:
        """
        Executa todos os três testes de estacionaridade.

        Args:
            data: Série temporal
            verbose: Se True, imprime resultados

        Returns:
            Dicionário com resultados de todos os testes
        """
        results = {
            'adf': self.adf_test(data),
            'kpss': self.kpss_test(data),
            'phillips_perron': self.phillips_perron_test(data)
        }

        # Decisão consolidada (maioria de 2/3)
        stationary_votes = sum([
            results['adf']['is_stationary'] if results['adf']['is_stationary'] is not None else False,
            results['kpss']['is_stationary'] if results['kpss']['is_stationary'] is not None else False,
            results['phillips_perron']['is_stationary'] if results['phillips_perron']['is_stationary'] is not None else False
        ])

        results['consensus'] = {
            'is_stationary': stationary_votes >= 2,
            'votes': f"{stationary_votes}/3",
            'recommendation': self._get_recommendation(results)
        }

        if verbose:
            self.print_results(results)

        self.results = results
        return results

    def _interpret_adf(self, p_value: float, statistic: float, critical_values: Dict) -> str:
        """Interpreta resultado do teste ADF."""
        interpretation = []

        if p_value < 0.01:
            interpretation.append("✓ Forte evidência de estacionaridade (p < 0.01)")
        elif p_value < 0.05:
            interpretation.append("✓ Evidência de estacionaridade (p < 0.05)")
        elif p_value < 0.10:
            interpretation.append("⚠ Evidência fraca de estacionaridade (p < 0.10)")
        else:
            interpretation.append("✗ Não há evidência de estacionaridade (p ≥ 0.10)")

        # Comparar com valores críticos
        for level, cv in critical_values.items():
            if statistic < cv:
                interpretation.append(f"  Estatística ({statistic:.4f}) < Valor Crítico {level} ({cv:.4f})")

        return "\n".join(interpretation)

    def _interpret_kpss(self, p_value: float, statistic: float, critical_values: Dict) -> str:
        """Interpreta resultado do teste KPSS (H0: estacionária)."""
        interpretation = []

        if p_value > 0.10:
            interpretation.append("✓ Forte evidência de estacionaridade (p > 0.10)")
        elif p_value > 0.05:
            interpretation.append("✓ Evidência de estacionaridade (p > 0.05)")
        elif p_value > 0.01:
            interpretation.append("⚠ Evidência fraca de estacionaridade (p > 0.01)")
        else:
            interpretation.append("✗ Não há evidência de estacionaridade (p ≤ 0.01)")

        return "\n".join(interpretation)

    def _interpret_pp(self, p_value: float, statistic: float, critical_values: Dict) -> str:
        """Interpreta resultado do teste Phillips-Perron."""
        interpretation = []

        if p_value < 0.01:
            interpretation.append("✓ Forte evidência de estacionaridade (p < 0.01)")
        elif p_value < 0.05:
            interpretation.append("✓ Evidência de estacionaridade (p < 0.05)")
        elif p_value < 0.10:
            interpretation.append("⚠ Evidência fraca de estacionaridade (p < 0.10)")
        else:
            interpretation.append("✗ Não há evidência de estacionaridade (p ≥ 0.10)")

        return "\n".join(interpretation)

    def _get_recommendation(self, results: Dict) -> str:
        """Gera recomendação baseada nos resultados."""
        adf_stat = results['adf'].get('is_stationary', None)
        kpss_stat = results['kpss'].get('is_stationary', None)
        pp_stat = results['phillips_perron'].get('is_stationary', None)

        if adf_stat and kpss_stat and pp_stat:
            return "✓ SÉRIE ESTACIONÁRIA: Todos os testes confirmam. Pode usar diretamente em modelos."
        elif not adf_stat and not kpss_stat and not pp_stat:
            return "✗ SÉRIE NÃO ESTACIONÁRIA: Todos os testes confirmam. Necessária diferenciação ou transformação."
        else:
            return "⚠ RESULTADO MISTO: Alguns testes divergem. Recomenda-se análise adicional ou aplicar diferenciação."

    def print_results(self, results: Optional[Dict] = None):
        """
        Imprime resultados dos testes de forma formatada.

        Args:
            results: Resultados dos testes (se None, usa self.results)
        """
        if results is None:
            results = self.results

        if not results:
            print("Nenhum resultado disponível. Execute run_all_tests() primeiro.")
            return

        print("\n" + "="*80)
        print("RESULTADOS DOS TESTES DE ESTACIONARIDADE")
        print("="*80)

        # Teste ADF
        print("\n1. TESTE ADF (Augmented Dickey-Fuller)")
        print("-" * 80)
        adf = results['adf']
        if 'error' not in adf:
            print(f"Estatística de teste: {adf['statistic']:.6f}")
            print(f"P-valor: {adf['p_value']:.6f}")
            print(f"Lags utilizados: {adf['used_lag']}")
            print(f"Observações: {adf['n_obs']}")
            print(f"Valores críticos:")
            for level, cv in adf['critical_values'].items():
                print(f"  {level}: {cv:.4f}")
            print(f"\nConclusão: {adf['conclusion']}")
            print(f"\n{adf['interpretation']}")
        else:
            print(f"Erro: {adf['error']}")

        # Teste KPSS
        print("\n2. TESTE KPSS (Kwiatkowski-Phillips-Schmidt-Shin)")
        print("-" * 80)
        kpss_result = results['kpss']
        if 'error' not in kpss_result:
            print(f"Estatística de teste: {kpss_result['statistic']:.6f}")
            print(f"P-valor: {kpss_result['p_value']:.6f}")
            print(f"Lags utilizados: {kpss_result['used_lag']}")
            print(f"Valores críticos:")
            for level, cv in kpss_result['critical_values'].items():
                print(f"  {level}: {cv:.4f}")
            print(f"\nConclusão: {kpss_result['conclusion']}")
            print(f"\n{kpss_result['interpretation']}")
        else:
            print(f"Erro: {kpss_result['error']}")

        # Teste Phillips-Perron
        print("\n3. TESTE PHILLIPS-PERRON")
        print("-" * 80)
        pp = results['phillips_perron']
        if 'error' not in pp:
            print(f"Estatística de teste: {pp['statistic']:.6f}")
            print(f"P-valor: {pp['p_value']:.6f}")
            print(f"Valores críticos:")
            for level, cv in pp['critical_values'].items():
                print(f"  {level}: {cv:.4f}")
            print(f"\nConclusão: {pp['conclusion']}")
            print(f"\n{pp['interpretation']}")
        else:
            print(f"Erro: {pp['error']}")

        # Consenso
        print("\n" + "="*80)
        print("CONSENSO DOS TESTES")
        print("="*80)
        consensus = results['consensus']
        print(f"Votos pela estacionaridade: {consensus['votes']}")
        print(f"Decisão: {'ESTACIONÁRIA' if consensus['is_stationary'] else 'NÃO ESTACIONÁRIA'}")
        print(f"\n{consensus['recommendation']}")
        print("="*80 + "\n")


def make_stationary(data: Union[np.ndarray, pd.Series],
                    max_diff: int = 2,
                    test_after: bool = True) -> Tuple[np.ndarray, int]:
    """
    Transforma uma série não estacionária em estacionária através de diferenciação.

    Args:
        data: Série temporal
        max_diff: Número máximo de diferenciações
        test_after: Se True, testa estacionaridade após cada diferenciação

    Returns:
        (série_estacionária, ordem_de_diferenciação)
    """
    tester = StationarityTests()

    # Converter para array
    if isinstance(data, pd.Series):
        original_data = data.values
    else:
        original_data = data.copy()

    current_data = original_data.copy()

    for d in range(max_diff + 1):
        if d > 0:
            current_data = np.diff(current_data)

        if test_after or d == 0:
            results = tester.run_all_tests(current_data, verbose=False)

            if results['consensus']['is_stationary']:
                print(f"Série tornou-se estacionária após {d} diferenciação(ões)")
                return current_data, d

    print(f"Atenção: Série não se tornou estacionária após {max_diff} diferenciações")
    return current_data, max_diff


# Exemplo de uso
if __name__ == "__main__":
    # Gerar série temporal de exemplo (não estacionária)
    np.random.seed(42)
    n = 200
    trend = np.linspace(0, 10, n)
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, n))
    noise = np.random.normal(0, 1, n)
    non_stationary_series = trend + seasonal + noise

    # Testar estacionaridade
    print("Testando série NÃO ESTACIONÁRIA (com tendência):")
    tester = StationarityTests()
    results = tester.run_all_tests(non_stationary_series)

    # Tornar estacionária
    print("\n\nAplicando diferenciação para tornar estacionária:")
    stationary_series, order = make_stationary(non_stationary_series)

    print("\n\nTestando série APÓS DIFERENCIAÇÃO:")
    results_after = tester.run_all_tests(stationary_series)
