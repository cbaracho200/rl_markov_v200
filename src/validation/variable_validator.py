"""
Módulo de Validação Integrada de Variáveis Preditoras
====================================================

Este módulo integra testes de estacionaridade e causalidade de Granger
para realizar validação completa de variáveis preditoras antes do treinamento.

Pipeline de validação:
1. Teste de estacionaridade (ADF, KPSS, Phillips-Perron)
2. Transformação para estacionaridade (se necessário)
3. Teste de causalidade de Granger
4. Seleção automática de preditores válidos

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from .stationarity_tests import StationarityTests, make_stationary
from .granger_causality import GrangerCausality
import warnings

warnings.filterwarnings('ignore')


class VariableValidator:
    """
    Validador integrado de variáveis preditoras.

    Realiza validação completa em 4 etapas:
    1. Verificação de estacionaridade
    2. Transformação (diferenciação) se necessário
    3. Teste de causalidade de Granger
    4. Seleção de preditores válidos
    """

    def __init__(self,
                 significance_level: float = 0.05,
                 max_lag_granger: int = 12,
                 min_causal_strength: str = 'weak',
                 auto_transform: bool = True):
        """
        Inicializa o validador.

        Args:
            significance_level: Nível de significância para testes (0.05 = 5%)
            max_lag_granger: Número máximo de lags para teste de Granger
            min_causal_strength: Força mínima de causalidade ('weak', 'moderate', 'strong', 'very_strong')
            auto_transform: Se True, transforma automaticamente séries não estacionárias
        """
        self.significance_level = significance_level
        self.max_lag_granger = max_lag_granger
        self.min_causal_strength = min_causal_strength
        self.auto_transform = auto_transform

        self.stationarity_tester = StationarityTests(significance_level)
        self.granger_tester = GrangerCausality(max_lag_granger, significance_level)

        self.validation_results = {}
        self.selected_predictors = []
        self.transformed_data = None

    def validate_all(self,
                     data: pd.DataFrame,
                     target_var: str,
                     predictor_vars: Optional[List[str]] = None,
                     verbose: bool = True) -> Dict:
        """
        Executa validação completa de variáveis preditoras.

        Args:
            data: DataFrame com todas as variáveis
            target_var: Nome da variável alvo
            predictor_vars: Lista de preditores candidatos (se None, usa todas exceto target)
            verbose: Se True, imprime resultados detalhados

        Returns:
            Dicionário com resultados completos da validação
        """
        if verbose:
            print("\n" + "=" * 80)
            print("VALIDAÇÃO INTEGRADA DE VARIÁVEIS PREDITORAS")
            print("=" * 80)
            print(f"Variável alvo: {target_var}")
            print(f"Nível de significância: {self.significance_level}")
            print(f"Força mínima de causalidade: {self.min_causal_strength}")
            print("=" * 80)

        # Selecionar variáveis preditoras candidatas
        if predictor_vars is None:
            predictor_vars = [col for col in data.columns if col != target_var]

        all_vars = [target_var] + predictor_vars

        # ETAPA 1: Teste de estacionaridade
        if verbose:
            print("\n" + "-" * 80)
            print("ETAPA 1: TESTES DE ESTACIONARIDADE")
            print("-" * 80)

        stationarity_results = self._test_stationarity(data[all_vars], verbose)

        # ETAPA 2: Transformação para estacionaridade
        if verbose:
            print("\n" + "-" * 80)
            print("ETAPA 2: TRANSFORMAÇÃO PARA ESTACIONARIDADE")
            print("-" * 80)

        transformed_data, transformation_info = self._transform_to_stationary(
            data[all_vars],
            stationarity_results,
            verbose
        )

        # ETAPA 3: Teste de causalidade de Granger
        if verbose:
            print("\n" + "-" * 80)
            print("ETAPA 3: TESTES DE CAUSALIDADE DE GRANGER")
            print("-" * 80)

        granger_results = self._test_granger_causality(
            transformed_data,
            target_var,
            predictor_vars,
            verbose
        )

        # ETAPA 4: Seleção de preditores
        if verbose:
            print("\n" + "-" * 80)
            print("ETAPA 4: SELEÇÃO DE PREDITORES VÁLIDOS")
            print("-" * 80)

        selected_predictors = self._select_valid_predictors(
            granger_results,
            stationarity_results,
            verbose
        )

        # Compilar resultados
        results = {
            'target_variable': target_var,
            'candidate_predictors': predictor_vars,
            'stationarity_results': stationarity_results,
            'transformation_info': transformation_info,
            'granger_results': granger_results,
            'selected_predictors': selected_predictors,
            'transformed_data': transformed_data,
            'validation_summary': self._create_summary(
                stationarity_results,
                granger_results,
                selected_predictors
            )
        }

        self.validation_results = results
        self.selected_predictors = selected_predictors
        self.transformed_data = transformed_data

        if verbose:
            self._print_final_summary(results)

        return results

    def _test_stationarity(self, data: pd.DataFrame, verbose: bool) -> Dict:
        """Testa estacionaridade de todas as variáveis."""
        results = {}

        for col in data.columns:
            if verbose:
                print(f"\nTestando: {col}")

            test_result = self.stationarity_tester.run_all_tests(data[col], verbose=False)
            is_stationary = test_result['consensus']['is_stationary']

            results[col] = {
                'is_stationary': is_stationary,
                'adf_p_value': test_result['adf'].get('p_value', None),
                'kpss_p_value': test_result['kpss'].get('p_value', None),
                'pp_p_value': test_result['phillips_perron'].get('p_value', None),
                'consensus': test_result['consensus']['votes'],
                'full_results': test_result
            }

            if verbose:
                status = "✓ ESTACIONÁRIA" if is_stationary else "✗ NÃO ESTACIONÁRIA"
                print(f"  {status} (consenso: {results[col]['consensus']})")

        return results

    def _transform_to_stationary(self,
                                 data: pd.DataFrame,
                                 stationarity_results: Dict,
                                 verbose: bool) -> Tuple[pd.DataFrame, Dict]:
        """Transforma séries não estacionárias."""
        transformed_data = data.copy()
        transformation_info = {}

        for col in data.columns:
            if not stationarity_results[col]['is_stationary']:
                if self.auto_transform:
                    if verbose:
                        print(f"\nTransformando: {col}")

                    # Aplicar diferenciação
                    original_series = data[col].values
                    stationary_series, diff_order = make_stationary(
                        original_series,
                        max_diff=2,
                        test_after=True
                    )

                    # Atualizar dados (ajustar tamanho devido à diferenciação)
                    # Preencher início com NaN
                    padded_series = np.concatenate([
                        np.full(diff_order, np.nan),
                        stationary_series
                    ])

                    # Ajustar tamanho para coincidir
                    if len(padded_series) > len(transformed_data):
                        padded_series = padded_series[:len(transformed_data)]
                    elif len(padded_series) < len(transformed_data):
                        padded_series = np.concatenate([
                            padded_series,
                            np.full(len(transformed_data) - len(padded_series), np.nan)
                        ])

                    transformed_data[col] = padded_series

                    transformation_info[col] = {
                        'transformed': True,
                        'method': 'differencing',
                        'order': diff_order
                    }

                    if verbose:
                        print(f"  ✓ Aplicada diferenciação de ordem {diff_order}")
                else:
                    if verbose:
                        print(f"  ⚠ {col} não é estacionária (auto_transform=False)")

                    transformation_info[col] = {
                        'transformed': False,
                        'warning': 'Série não estacionária não foi transformada'
                    }
            else:
                transformation_info[col] = {
                    'transformed': False,
                    'reason': 'Série já é estacionária'
                }

        # Remover linhas com NaN criadas pela diferenciação
        transformed_data = transformed_data.dropna()

        if verbose:
            print(f"\nDados após transformação: {len(transformed_data)} observações")

        return transformed_data, transformation_info

    def _test_granger_causality(self,
                                data: pd.DataFrame,
                                target_var: str,
                                predictor_vars: List[str],
                                verbose: bool) -> Dict:
        """Testa causalidade de Granger."""
        results = self.granger_tester.test_all_combinations(
            data,
            variables=[target_var] + predictor_vars,
            target_var=target_var,
            verbose=verbose
        )

        return results

    def _select_valid_predictors(self,
                                 granger_results: Dict,
                                 stationarity_results: Dict,
                                 verbose: bool) -> List[Dict]:
        """Seleciona preditores válidos baseado em todos os critérios."""
        valid_predictors = []

        strength_order = ['weak', 'moderate', 'strong', 'very_strong']
        min_level = strength_order.index(self.min_causal_strength)

        for result in granger_results['significant_relationships']:
            predictor = result['x_var']
            strength = result.get('strength', 'none')

            # Verificar força de causalidade
            if strength in strength_order and strength_order.index(strength) >= min_level:
                valid_predictors.append({
                    'variable': predictor,
                    'p_value': result['best_p_value'],
                    'lag': result['best_lag'],
                    'strength': strength,
                    'f_statistic': result['best_f_stat'],
                    'is_stationary': stationarity_results[predictor]['is_stationary'],
                    'was_transformed': predictor in self.validation_results.get('transformation_info', {})
                })

        # Ordenar por p-value (mais significativos primeiro)
        valid_predictors = sorted(valid_predictors, key=lambda x: x['p_value'])

        if verbose:
            if valid_predictors:
                print(f"\n✓ {len(valid_predictors)} preditores válidos selecionados:")
                for i, pred in enumerate(valid_predictors, 1):
                    print(f"\n{i}. {pred['variable']}")
                    print(f"   Causalidade de Granger: p = {pred['p_value']:.6f} ({pred['strength']})")
                    print(f"   Lag ótimo: {pred['lag']}")
                    print(f"   Estatística F: {pred['f_statistic']:.4f}")
                    print(f"   Estacionária: {'Sim' if pred['is_stationary'] else 'Transformada'}")
            else:
                print("\n✗ Nenhum preditor válido encontrado!")
                print("   Considere:")
                print("   - Reduzir min_causal_strength")
                print("   - Aumentar max_lag_granger")
                print("   - Verificar qualidade dos dados")

        return valid_predictors

    def _create_summary(self,
                        stationarity_results: Dict,
                        granger_results: Dict,
                        selected_predictors: List[Dict]) -> Dict:
        """Cria resumo da validação."""
        n_vars = len(stationarity_results)
        n_stationary = sum(1 for r in stationarity_results.values() if r['is_stationary'])
        n_transformed = n_vars - n_stationary

        n_significant_causal = len(granger_results['significant_relationships'])
        n_selected = len(selected_predictors)

        return {
            'total_variables': n_vars,
            'stationary_variables': n_stationary,
            'transformed_variables': n_transformed,
            'significant_causal_relationships': n_significant_causal,
            'selected_predictors': n_selected,
            'selection_rate': f"{n_selected}/{n_vars-1}" if n_vars > 1 else "N/A"
        }

    def _print_final_summary(self, results: Dict):
        """Imprime resumo final da validação."""
        summary = results['validation_summary']

        print("\n" + "=" * 80)
        print("RESUMO FINAL DA VALIDAÇÃO")
        print("=" * 80)
        print(f"Variável alvo: {results['target_variable']}")
        print(f"Total de variáveis testadas: {summary['total_variables']}")
        print(f"Variáveis estacionárias: {summary['stationary_variables']}")
        print(f"Variáveis transformadas: {summary['transformed_variables']}")
        print(f"Relações causais significativas: {summary['significant_causal_relationships']}")
        print(f"Preditores selecionados: {summary['selected_predictors']} ({summary['selection_rate']})")

        if summary['selected_predictors'] > 0:
            print("\n✓ VALIDAÇÃO CONCLUÍDA COM SUCESSO")
            print("\nPreditores recomendados (ordenados por significância):")
            for i, pred in enumerate(results['selected_predictors'], 1):
                print(f"  {i}. {pred['variable']} (p={pred['p_value']:.6f}, lag={pred['lag']})")
        else:
            print("\n⚠ ATENÇÃO: Nenhum preditor válido encontrado")

        print("=" * 80 + "\n")

    def get_validated_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Retorna dados transformados e lista de preditores válidos.

        Returns:
            (dados_transformados, lista_de_preditores)
        """
        if self.transformed_data is None:
            raise ValueError("Execute validate_all() primeiro")

        predictor_names = [p['variable'] for p in self.selected_predictors]
        return self.transformed_data, predictor_names

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna DataFrame com importância das features baseada em causalidade.

        Returns:
            DataFrame com rankings de importância
        """
        if not self.selected_predictors:
            return pd.DataFrame()

        importance_data = []
        for pred in self.selected_predictors:
            importance_data.append({
                'Variable': pred['variable'],
                'P_Value': pred['p_value'],
                'Importance_Score': 1 - pred['p_value'],  # Score de importância
                'Lag': pred['lag'],
                'Strength': pred['strength'],
                'F_Statistic': pred['f_statistic']
            })

        df = pd.DataFrame(importance_data)
        df = df.sort_values('Importance_Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)

        return df[['Rank', 'Variable', 'Importance_Score', 'P_Value', 'Strength', 'Lag', 'F_Statistic']]


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados de exemplo
    np.random.seed(42)
    n = 250

    # Variável alvo Y com múltiplos preditores
    x1 = np.random.normal(0, 1, n)  # Preditor forte
    x2 = np.random.normal(0, 1, n)  # Preditor moderado
    x3 = np.random.normal(0, 1, n)  # Não é preditor
    trend = np.linspace(0, 5, n)    # Tendência

    y = np.zeros(n)
    for t in range(3, n):
        y[t] = (0.6 * x1[t-1] +         # X1 causa Y (lag 1)
                0.3 * x2[t-2] +         # X2 causa Y (lag 2)
                0.2 * y[t-1] +          # Autocorrelação
                0.1 * trend[t] +        # Tendência
                np.random.normal(0, 0.5))

    # Criar DataFrame
    data = pd.DataFrame({
        'Y': y,
        'X1': x1,
        'X2': x2,
        'X3': x3
    })

    print("EXEMPLO: Validação Integrada de Variáveis Preditoras")
    print("=" * 80)
    print("Dados gerados:")
    print("  - Y: variável alvo (com tendência)")
    print("  - X1: preditor forte (lag 1)")
    print("  - X2: preditor moderado (lag 2)")
    print("  - X3: não é preditor (ruído)")
    print("=" * 80)

    # Executar validação completa
    validator = VariableValidator(
        significance_level=0.05,
        max_lag_granger=10,
        min_causal_strength='weak',
        auto_transform=True
    )

    results = validator.validate_all(
        data=data,
        target_var='Y',
        verbose=True
    )

    # Obter importância das features
    print("\n" + "=" * 80)
    print("IMPORTÂNCIA DAS FEATURES")
    print("=" * 80)
    importance = validator.get_feature_importance()
    print(importance.to_string(index=False))
