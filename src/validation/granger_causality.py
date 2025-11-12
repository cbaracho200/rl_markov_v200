"""
Módulo de Testes de Causalidade de Granger
==========================================

Este módulo implementa testes de causalidade de Granger para identificar relações
de precedência temporal entre variáveis, essencial para seleção de variáveis preditoras.

Conceito:
- X "Granger-causa" Y se valores passados de X contêm informação útil para prever Y
- Não implica causalidade real, apenas precedência temporal

Testes implementados:
- Teste de Granger bivariado (X → Y)
- Teste de Granger multivariado (múltiplas variáveis)
- Seleção automática de variáveis preditoras

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class GrangerCausality:
    """
    Classe para realizar testes de causalidade de Granger.

    O teste de Granger avalia se valores passados de uma variável X ajudam
    a prever valores futuros de uma variável Y, além da informação contida
    nos próprios valores passados de Y.

    H0: X NÃO Granger-causa Y
    H1: X Granger-causa Y

    Se p-value < 0.05, rejeitamos H0 e concluímos que X Granger-causa Y.
    """

    def __init__(self, max_lag: int = 12, significance_level: float = 0.05):
        """
        Inicializa o testador de causalidade de Granger.

        Args:
            max_lag: Número máximo de lags a testar
            significance_level: Nível de significância (default: 0.05 = 5%)
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.results = {}

    def test_granger_causality(self,
                               data: pd.DataFrame,
                               x_var: str,
                               y_var: str,
                               max_lag: Optional[int] = None,
                               verbose: bool = False) -> Dict:
        """
        Testa se x_var Granger-causa y_var.

        Args:
            data: DataFrame com as variáveis
            x_var: Nome da variável preditora (causa)
            y_var: Nome da variável alvo (efeito)
            max_lag: Número máximo de lags (se None, usa self.max_lag)
            verbose: Se True, imprime resultados detalhados

        Returns:
            Dicionário com resultados do teste
        """
        if max_lag is None:
            max_lag = self.max_lag

        try:
            # Preparar dados (y, x)
            test_data = data[[y_var, x_var]].dropna()

            if len(test_data) < max_lag + 10:
                return {
                    'x_var': x_var,
                    'y_var': y_var,
                    'granger_causes': False,
                    'error': 'Dados insuficientes para o teste'
                }

            # Executar teste de Granger para múltiplos lags
            gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

            # Extrair p-values de cada lag
            p_values = {}
            f_stats = {}
            best_lag = None
            best_p_value = 1.0

            for lag in range(1, max_lag + 1):
                # Usar teste F
                f_test = gc_result[lag][0]['ssr_ftest']
                f_stat = f_test[0]
                p_value = f_test[1]

                p_values[lag] = p_value
                f_stats[lag] = f_stat

                # Encontrar melhor lag (menor p-value)
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_lag = lag

            # Determinar se há causalidade de Granger
            granger_causes = best_p_value < self.significance_level

            result = {
                'x_var': x_var,
                'y_var': y_var,
                'relationship': f"{x_var} → {y_var}",
                'granger_causes': granger_causes,
                'best_lag': best_lag,
                'best_p_value': best_p_value,
                'best_f_stat': f_stats[best_lag],
                'all_p_values': p_values,
                'all_f_stats': f_stats,
                'significance_level': self.significance_level,
                'conclusion': self._interpret_result(x_var, y_var, granger_causes, best_lag, best_p_value),
                'strength': self._assess_strength(best_p_value)
            }

            if verbose:
                self._print_test_result(result)

            return result

        except Exception as e:
            return {
                'x_var': x_var,
                'y_var': y_var,
                'granger_causes': False,
                'error': str(e)
            }

    def test_all_combinations(self,
                              data: pd.DataFrame,
                              variables: Optional[List[str]] = None,
                              target_var: Optional[str] = None,
                              verbose: bool = True) -> Dict:
        """
        Testa causalidade de Granger para todas as combinações de variáveis.

        Args:
            data: DataFrame com as variáveis
            variables: Lista de variáveis a testar (se None, usa todas as colunas)
            target_var: Se especificado, testa apenas X → target_var
            verbose: Se True, imprime progresso

        Returns:
            Dicionário com todos os resultados
        """
        if variables is None:
            variables = data.columns.tolist()

        results = {
            'all_tests': [],
            'significant_relationships': [],
            'causal_matrix': None
        }

        if target_var is not None:
            # Testar apenas X → target_var
            if verbose:
                print(f"\nTestando causalidade de Granger para variável alvo: {target_var}")
                print("=" * 80)

            predictor_vars = [v for v in variables if v != target_var]

            for x_var in predictor_vars:
                if verbose:
                    print(f"\nTestando: {x_var} → {target_var}")

                result = self.test_granger_causality(data, x_var, target_var, verbose=False)
                results['all_tests'].append(result)

                if result.get('granger_causes', False):
                    results['significant_relationships'].append(result)

                    if verbose:
                        print(f"  ✓ {x_var} Granger-causa {target_var}")
                        print(f"    Lag ótimo: {result['best_lag']}, p-value: {result['best_p_value']:.6f}")
                        print(f"    Força: {result['strength']}")
                else:
                    if verbose:
                        print(f"  ✗ {x_var} NÃO Granger-causa {target_var}")
                        print(f"    p-value: {result.get('best_p_value', 'N/A')}")

        else:
            # Testar todas as combinações bidirecionais
            if verbose:
                print(f"\nTestando todas as combinações de causalidade de Granger")
                print(f"Variáveis: {variables}")
                print("=" * 80)

            for x_var, y_var in combinations(variables, 2):
                # Testar X → Y
                if verbose:
                    print(f"\nTestando: {x_var} → {y_var}")

                result_xy = self.test_granger_causality(data, x_var, y_var, verbose=False)
                results['all_tests'].append(result_xy)

                if result_xy.get('granger_causes', False):
                    results['significant_relationships'].append(result_xy)
                    if verbose:
                        print(f"  ✓ SIGNIFICANTE (p={result_xy['best_p_value']:.6f})")

                # Testar Y → X (bidirecional)
                if verbose:
                    print(f"Testando: {y_var} → {x_var}")

                result_yx = self.test_granger_causality(data, y_var, x_var, verbose=False)
                results['all_tests'].append(result_yx)

                if result_yx.get('granger_causes', False):
                    results['significant_relationships'].append(result_yx)
                    if verbose:
                        print(f"  ✓ SIGNIFICANTE (p={result_yx['best_p_value']:.6f})")

        # Criar matriz de causalidade
        results['causal_matrix'] = self._create_causal_matrix(results['all_tests'], variables)

        if verbose:
            print("\n" + "=" * 80)
            print(f"RESUMO: {len(results['significant_relationships'])} relações causais encontradas")
            print("=" * 80)

        self.results = results
        return results

    def select_predictors(self,
                          data: pd.DataFrame,
                          target_var: str,
                          min_strength: str = 'weak',
                          verbose: bool = True) -> List[str]:
        """
        Seleciona variáveis preditoras usando causalidade de Granger.

        Args:
            data: DataFrame com as variáveis
            target_var: Variável alvo a ser prevista
            min_strength: Força mínima ('weak', 'moderate', 'strong', 'very_strong')
            verbose: Se True, imprime resultados

        Returns:
            Lista de variáveis preditoras selecionadas
        """
        # Testar todas as variáveis
        variables = [col for col in data.columns if col != target_var]
        results = self.test_all_combinations(data, variables, target_var, verbose=False)

        # Filtrar por força
        strength_order = ['weak', 'moderate', 'strong', 'very_strong']
        min_level = strength_order.index(min_strength)

        selected = []
        for result in results['significant_relationships']:
            strength = result.get('strength', 'none')
            if strength in strength_order and strength_order.index(strength) >= min_level:
                selected.append({
                    'variable': result['x_var'],
                    'p_value': result['best_p_value'],
                    'lag': result['best_lag'],
                    'strength': strength
                })

        # Ordenar por p-value
        selected = sorted(selected, key=lambda x: x['p_value'])

        if verbose:
            print("\n" + "=" * 80)
            print(f"SELEÇÃO DE PREDITORES PARA: {target_var}")
            print("=" * 80)
            print(f"Força mínima: {min_strength}")
            print(f"\nVariáveis selecionadas: {len(selected)}")
            print("-" * 80)

            for i, pred in enumerate(selected, 1):
                print(f"{i}. {pred['variable']}")
                print(f"   P-value: {pred['p_value']:.6f}")
                print(f"   Lag ótimo: {pred['lag']}")
                print(f"   Força: {pred['strength']}")

            print("=" * 80)

        return [p['variable'] for p in selected]

    def _interpret_result(self, x_var: str, y_var: str, causes: bool, lag: int, p_value: float) -> str:
        """Interpreta o resultado do teste."""
        if causes:
            return f"✓ {x_var} Granger-causa {y_var} (lag={lag}, p={p_value:.6f})"
        else:
            return f"✗ {x_var} NÃO Granger-causa {y_var} (p={p_value:.6f})"

    def _assess_strength(self, p_value: float) -> str:
        """Avalia a força da relação causal."""
        if p_value < 0.001:
            return 'very_strong'
        elif p_value < 0.01:
            return 'strong'
        elif p_value < 0.05:
            return 'moderate'
        elif p_value < 0.10:
            return 'weak'
        else:
            return 'none'

    def _create_causal_matrix(self, test_results: List[Dict], variables: List[str]) -> pd.DataFrame:
        """
        Cria matriz de causalidade (linhas causam colunas).

        Args:
            test_results: Lista de resultados dos testes
            variables: Lista de variáveis

        Returns:
            DataFrame matriz NxN onde [i,j] indica se i causa j
        """
        matrix = pd.DataFrame(0, index=variables, columns=variables)

        for result in test_results:
            if 'error' not in result:
                x = result['x_var']
                y = result['y_var']
                if x in variables and y in variables:
                    if result.get('granger_causes', False):
                        # Usar 1 - p_value como "força" da causalidade
                        matrix.loc[x, y] = 1 - result['best_p_value']

        return matrix

    def _print_test_result(self, result: Dict):
        """Imprime resultado de um teste individual."""
        print("\n" + "-" * 80)
        print(f"Teste de Causalidade de Granger: {result['relationship']}")
        print("-" * 80)
        print(f"Conclusão: {result['conclusion']}")
        print(f"Melhor lag: {result['best_lag']}")
        print(f"P-value: {result['best_p_value']:.6f}")
        print(f"Estatística F: {result['best_f_stat']:.4f}")
        print(f"Força da relação: {result['strength']}")

        print("\nP-values por lag:")
        for lag, p_val in result['all_p_values'].items():
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  Lag {lag}: {p_val:.6f} {sig}")

    def plot_causal_network(self, threshold: float = 0.05):
        """
        Plota rede de causalidade (requer networkx e matplotlib).

        Args:
            threshold: P-value máximo para incluir na rede
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            if not self.results:
                print("Execute test_all_combinations() primeiro.")
                return

            # Criar grafo direcionado
            G = nx.DiGraph()

            # Adicionar arestas significativas
            for result in self.results['significant_relationships']:
                if result['best_p_value'] < threshold:
                    weight = 1 - result['best_p_value']  # Converter p-value em peso
                    G.add_edge(
                        result['x_var'],
                        result['y_var'],
                        weight=weight,
                        lag=result['best_lag'],
                        p_value=result['best_p_value']
                    )

            if len(G.edges()) == 0:
                print("Nenhuma relação causal significativa encontrada.")
                return

            # Plotar
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Desenhar nós
            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                   node_size=3000, alpha=0.9)

            # Desenhar arestas com espessura proporcional ao peso
            edges = G.edges()
            weights = [G[u][v]['weight'] * 5 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6,
                                   edge_color='gray', arrows=True,
                                   arrowsize=20, arrowstyle='->')

            # Labels dos nós
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            # Labels das arestas (lags)
            edge_labels = {(u, v): f"lag={G[u][v]['lag']}"
                          for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

            plt.title("Rede de Causalidade de Granger", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Para plotar a rede, instale: pip install networkx matplotlib")


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo com relações causais
    np.random.seed(42)
    n = 200

    # X1 → Y (X1 causa Y)
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = np.zeros(n)

    for t in range(2, n):
        # Y depende de valores passados de X1 (lag 1 e 2)
        y[t] = 0.5 * x1[t-1] + 0.3 * x1[t-2] + 0.2 * y[t-1] + np.random.normal(0, 0.5)

    # Criar DataFrame
    data = pd.DataFrame({
        'Y': y,
        'X1': x1,
        'X2': x2
    })

    print("EXEMPLO: Teste de Causalidade de Granger")
    print("=" * 80)
    print("Dados gerados: X1 → Y (causal), X2 (não causal)")
    print("=" * 80)

    # Testar causalidade
    gc = GrangerCausality(max_lag=10)

    # Teste individual
    print("\n1. TESTE INDIVIDUAL: X1 → Y")
    result1 = gc.test_granger_causality(data, 'X1', 'Y', verbose=True)

    print("\n2. TESTE INDIVIDUAL: X2 → Y")
    result2 = gc.test_granger_causality(data, 'X2', 'Y', verbose=True)

    # Seleção de preditores
    print("\n3. SELEÇÃO AUTOMÁTICA DE PREDITORES")
    predictors = gc.select_predictors(data, 'Y', min_strength='weak', verbose=True)
