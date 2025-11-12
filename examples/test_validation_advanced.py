"""
Exemplo Avançado: Validação de Variáveis Preditoras + Modelos Avançados
=======================================================================

Este exemplo demonstra o pipeline completo de validação estatística de variáveis
preditoras e uso de modelos avançados (SARIMA, SARIMAX, VAR).

Pipeline:
1. Geração de dados sintéticos com relações causais
2. Testes de estacionaridade (ADF, KPSS, Phillips-Perron)
3. Transformação para estacionaridade (diferenciação)
4. Testes de causalidade de Granger
5. Seleção automática de preditores válidos
6. Treinamento de modelos avançados com preditores validados
7. Comparação de desempenho

Tempo estimado: 5-10 minutos

Autor: Advanced RL Framework
Nível: PhD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos de validação
from src.validation import (
    StationarityTests,
    GrangerCausality,
    VariableValidator
)

# Importar modelos avançados
from src.models import (
    SARIMAPredictor,
    SARIMAXPredictor,
    VARPredictor
)


def generate_economic_data(n=300, seed=42):
    """
    Gera dados econômicos sintéticos com relações causais realistas.

    Variáveis:
    - GDP: PIB (variável alvo)
    - Interest_Rate: Taxa de juros (causa GDP)
    - Inflation: Inflação (causa GDP)
    - Unemployment: Desemprego (causa GDP)
    - Consumer_Confidence: Confiança do consumidor (causa GDP)
    - Random_Noise: Ruído aleatório (não causa GDP)

    Returns:
        DataFrame com todas as variáveis
    """
    np.random.seed(seed)

    # Variáveis exógenas
    interest_rate = 5 + 2 * np.sin(2 * np.pi * np.arange(n) / 48) + np.random.normal(0, 0.5, n)
    inflation = 3 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 36 + np.pi/4) + np.random.normal(0, 0.3, n)
    unemployment = 7 - 2 * np.sin(2 * np.pi * np.arange(n) / 60) + np.random.normal(0, 0.8, n)
    consumer_conf = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 5, n)
    random_noise = np.random.normal(0, 1, n)  # Não deve ser selecionado

    # GDP (variável alvo) - depende causalmente das outras
    gdp = np.zeros(n)
    trend = np.linspace(100, 150, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)  # Sazonalidade anual

    for t in range(3, n):
        gdp[t] = (trend[t] +
                  seasonal[t] +
                  0.5 * gdp[t-1] +                    # Autocorrelação
                  -1.2 * interest_rate[t-1] +          # Taxa de juros causa GDP (lag 1)
                  0.8 * inflation[t-2] +               # Inflação causa GDP (lag 2)
                  -0.6 * unemployment[t-1] +           # Desemprego causa GDP (lag 1)
                  0.15 * consumer_conf[t-1] +          # Confiança causa GDP (lag 1)
                  np.random.normal(0, 2))

    # Criar DataFrame
    data = pd.DataFrame({
        'GDP': gdp,
        'Interest_Rate': interest_rate,
        'Inflation': inflation,
        'Unemployment': unemployment,
        'Consumer_Confidence': consumer_conf,
        'Random_Noise': random_noise
    })

    return data


def print_section_header(title):
    """Imprime cabeçalho de seção formatado."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def main():
    """Executa exemplo completo de validação e modelagem avançada."""

    print_section_header("EXEMPLO AVANÇADO: VALIDAÇÃO DE VARIÁVEIS + MODELOS AVANÇADOS")

    # ==========================================================================
    # ETAPA 1: Gerar Dados
    # ==========================================================================
    print_section_header("ETAPA 1: GERAÇÃO DE DADOS ECONÔMICOS SINTÉTICOS")

    data = generate_economic_data(n=300)

    print(f"\nDados gerados: {len(data)} observações mensais (~25 anos)")
    print(f"Variáveis: {list(data.columns)}")
    print("\nEstatísticas descritivas:")
    print(data.describe().round(2))

    print("\nRelações causais REAIS (injetadas nos dados):")
    print("  - Interest_Rate → GDP (lag 1, coef = -1.2)")
    print("  - Inflation → GDP (lag 2, coef = 0.8)")
    print("  - Unemployment → GDP (lag 1, coef = -0.6)")
    print("  - Consumer_Confidence → GDP (lag 1, coef = 0.15)")
    print("  - Random_Noise → GDP (nenhuma relação, não deve ser selecionado)")

    # Dividir em treino e teste
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"\nDivisão: Treino = {len(train_data)} | Teste = {len(test_data)}")

    # ==========================================================================
    # ETAPA 2: Validação Completa de Variáveis
    # ==========================================================================
    print_section_header("ETAPA 2: VALIDAÇÃO INTEGRADA DE VARIÁVEIS PREDITORAS")

    print("\nInicializando VariableValidator...")
    print("  - Nível de significância: 0.05 (5%)")
    print("  - Máximo de lags (Granger): 12")
    print("  - Força mínima de causalidade: 'weak'")
    print("  - Transformação automática: Ativada")

    validator = VariableValidator(
        significance_level=0.05,
        max_lag_granger=12,
        min_causal_strength='weak',
        auto_transform=True
    )

    print("\nExecutando validação completa (4 etapas)...")
    print("  1. Testes de estacionaridade (ADF, KPSS, Phillips-Perron)")
    print("  2. Transformação para estacionaridade (se necessário)")
    print("  3. Testes de causalidade de Granger")
    print("  4. Seleção de preditores válidos")

    validation_results = validator.validate_all(
        data=train_data,
        target_var='GDP',
        verbose=True
    )

    # Obter preditores selecionados
    selected_predictors = [p['variable'] for p in validation_results['selected_predictors']]

    print_section_header("PREDITORES SELECIONADOS PELA VALIDAÇÃO")
    print(f"\nTotal: {len(selected_predictors)} preditores válidos")

    if selected_predictors:
        print("\nRanking de importância:")
        importance_df = validator.get_feature_importance()
        print(importance_df.to_string(index=False))

        # Comparar com relações reais
        expected_predictors = {'Interest_Rate', 'Inflation', 'Unemployment', 'Consumer_Confidence'}
        selected_set = set(selected_predictors)

        print("\nValidação dos resultados:")
        print(f"  ✓ Preditores corretos identificados: {len(selected_set & expected_predictors)}/{len(expected_predictors)}")
        print(f"  ✗ Falsos positivos: {len(selected_set - expected_predictors)}")
        print(f"  ✗ Falsos negativos: {len(expected_predictors - selected_set)}")

        if 'Random_Noise' in selected_predictors:
            print("  ⚠ AVISO: Random_Noise foi incorretamente selecionado (pode acontecer com 5% de chance)")
        else:
            print("  ✓ Random_Noise corretamente rejeitado")
    else:
        print("\n⚠ AVISO: Nenhum preditor válido foi encontrado!")
        print("  Continuando com modelo univariado SARIMA...")

    # ==========================================================================
    # ETAPA 3: Treinamento de Modelos Avançados
    # ==========================================================================
    print_section_header("ETAPA 3: TREINAMENTO DE MODELOS AVANÇADOS")

    # Obter dados transformados
    transformed_data, _ = validator.get_validated_data()

    # Preparar dados para modelagem
    y_train = transformed_data['GDP']
    y_test_original = test_data['GDP'].iloc[:len(test_data)]

    results = {}

    # --- Modelo 1: SARIMA (univariado, baseline) ---
    print("\n" + "-" * 80)
    print("MODELO 1: SARIMA (univariado)")
    print("-" * 80)
    print("Configuração: SARIMA(1,1,1)(1,1,1,12)")
    print("Descrição: Modelo baseline sem preditores externos")

    try:
        sarima = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )

        print("\nTreinando...")
        sarima.fit(train_data['GDP'])

        print("\nFazendo previsão...")
        forecast_sarima = sarima.predict(steps=len(test_data))

        # Avaliar
        actual = test_data['GDP'].values
        mape_sarima = np.mean(np.abs((actual - forecast_sarima) / (actual + 1e-8))) * 100
        rmse_sarima = np.sqrt(np.mean((actual - forecast_sarima) ** 2))

        results['SARIMA'] = {
            'forecast': forecast_sarima,
            'mape': mape_sarima,
            'rmse': rmse_sarima
        }

        print(f"\n✓ Modelo treinado com sucesso!")
        print(f"  MAPE: {mape_sarima:.2f}%")
        print(f"  RMSE: {rmse_sarima:.4f}")

        # Critérios de informação
        criteria = sarima.get_information_criteria()
        print(f"  AIC: {criteria['AIC']:.2f}")
        print(f"  BIC: {criteria['BIC']:.2f}")

    except Exception as e:
        print(f"\n✗ Erro no SARIMA: {str(e)}")

    # --- Modelo 2: SARIMAX (com preditores validados) ---
    if selected_predictors:
        print("\n" + "-" * 80)
        print("MODELO 2: SARIMAX (com preditores validados)")
        print("-" * 80)
        print(f"Configuração: SARIMAX(1,1,1)(1,1,1,12) + {len(selected_predictors)} preditores")
        print(f"Preditores: {selected_predictors}")

        try:
            # Preparar variáveis exógenas
            exog_train = train_data[selected_predictors]
            exog_test = test_data[selected_predictors]

            sarimax = SARIMAXPredictor(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                exog_names=selected_predictors
            )

            print("\nTreinando com variáveis exógenas...")
            sarimax.fit(train_data['GDP'], exog=exog_train)

            print("\nFazendo previsão...")
            forecast_sarimax = sarimax.predict(steps=len(test_data), exog=exog_test)

            # Avaliar
            mape_sarimax = np.mean(np.abs((actual - forecast_sarimax) / (actual + 1e-8))) * 100
            rmse_sarimax = np.sqrt(np.mean((actual - forecast_sarimax) ** 2))

            results['SARIMAX'] = {
                'forecast': forecast_sarimax,
                'mape': mape_sarimax,
                'rmse': rmse_sarimax
            }

            print(f"\n✓ Modelo treinado com sucesso!")
            print(f"  MAPE: {mape_sarimax:.2f}%")
            print(f"  RMSE: {rmse_sarimax:.4f}")

            # Coeficientes das exógenas
            coeffs = sarimax.get_exog_coefficients()
            print("\n  Coeficientes das variáveis exógenas:")
            for var, coef in coeffs.items():
                print(f"    {var}: {coef:.4f}")

            # Critérios de informação
            criteria = sarimax.get_information_criteria()
            print(f"\n  AIC: {criteria['AIC']:.2f}")
            print(f"  BIC: {criteria['BIC']:.2f}")

        except Exception as e:
            print(f"\n✗ Erro no SARIMAX: {str(e)}")

    # --- Modelo 3: VAR (multivariado) ---
    print("\n" + "-" * 80)
    print("MODELO 3: VAR (Vector Autoregression)")
    print("-" * 80)

    if selected_predictors:
        var_variables = ['GDP'] + selected_predictors[:3]  # Usar top 3 preditores
        print(f"Configuração: VAR com seleção automática de lag")
        print(f"Variáveis: {var_variables}")

        try:
            var_data_train = train_data[var_variables]
            var_data_test = test_data[var_variables]

            var = VARPredictor(maxlags=None, ic='aic')

            print("\nTreinando VAR...")
            var.fit(var_data_train)

            print("\nFazendo previsão para GDP...")
            forecast_var = var.predict_single_variable('GDP', steps=len(test_data))

            # Avaliar
            mape_var = np.mean(np.abs((actual - forecast_var) / (actual + 1e-8))) * 100
            rmse_var = np.sqrt(np.mean((actual - forecast_var) ** 2))

            results['VAR'] = {
                'forecast': forecast_var,
                'mape': mape_var,
                'rmse': rmse_var
            }

            print(f"\n✓ Modelo treinado com sucesso!")
            print(f"  MAPE: {mape_var:.2f}%")
            print(f"  RMSE: {rmse_var:.4f}")

            # Critérios de informação
            criteria = var.get_information_criteria()
            print(f"  AIC: {criteria['AIC']:.4f}")
            print(f"  BIC: {criteria['BIC']:.4f}")

            # Matriz de causalidade
            print("\n  Matriz de Causalidade de Granger (p-values):")
            causality_matrix = var.get_granger_causality_matrix()
            print(causality_matrix.round(4))

        except Exception as e:
            print(f"\n✗ Erro no VAR: {str(e)}")
    else:
        print("Pulando VAR (nenhum preditor válido encontrado)")

    # ==========================================================================
    # ETAPA 4: Comparação de Desempenho
    # ==========================================================================
    print_section_header("ETAPA 4: COMPARAÇÃO DE DESEMPENHO")

    if results:
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Modelo': model_name,
                'MAPE (%)': result['mape'],
                'RMSE': result['rmse']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAPE (%)')

        print("\nRanking de modelos (por MAPE):")
        print(comparison_df.to_string(index=False))

        # Identificar melhor modelo
        best_model = comparison_df.iloc[0]['Modelo']
        best_mape = comparison_df.iloc[0]['MAPE (%)']

        print(f"\n🏆 MELHOR MODELO: {best_model}")
        print(f"   MAPE: {best_mape:.2f}%")

        # Análise de melhoria
        if 'SARIMA' in results and 'SARIMAX' in results:
            improvement = ((results['SARIMA']['mape'] - results['SARIMAX']['mape']) /
                          results['SARIMA']['mape'] * 100)
            print(f"\n📊 Melhoria do SARIMAX sobre SARIMA: {improvement:.2f}%")
            if improvement > 0:
                print("   ✓ Preditores externos melhoraram a previsão!")
            else:
                print("   ⚠ Preditores externos não melhoraram a previsão")

        # ==========================================================================
        # ETAPA 5: Visualização
        # ==========================================================================
        print_section_header("ETAPA 5: VISUALIZAÇÃO DOS RESULTADOS")

        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # Plot 1: Previsões vs Valores Reais
            ax1 = axes[0]
            ax1.plot(actual, 'k-', label='Real', linewidth=2, alpha=0.7)

            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (model_name, result) in enumerate(results.items()):
                ax1.plot(result['forecast'], '--', label=f'{model_name} (MAPE: {result["mape"]:.2f}%)',
                        linewidth=1.5, alpha=0.8, color=colors[i % len(colors)])

            ax1.set_xlabel('Período de Teste', fontsize=12)
            ax1.set_ylabel('GDP', fontsize=12)
            ax1.set_title('Comparação de Previsões vs Valores Reais', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Erros absolutos
            ax2 = axes[1]
            for i, (model_name, result) in enumerate(results.items()):
                errors = np.abs(actual - result['forecast'])
                ax2.plot(errors, '-', label=f'{model_name}', linewidth=1.5, alpha=0.8,
                        color=colors[i % len(colors)])

            ax2.set_xlabel('Período de Teste', fontsize=12)
            ax2.set_ylabel('Erro Absoluto', fontsize=12)
            ax2.set_title('Erros Absolutos de Previsão', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
            print("\n✓ Gráfico salvo como 'validation_results.png'")

        except Exception as e:
            print(f"\n⚠ Não foi possível criar visualização: {str(e)}")

    # ==========================================================================
    # CONCLUSÃO
    # ==========================================================================
    print_section_header("CONCLUSÃO")

    print("\n✓ Pipeline completo executado com sucesso!")
    print("\nResumo:")
    print(f"  1. {len(selected_predictors)} preditores válidos identificados")
    print(f"  2. {len(results)} modelos avançados treinados")
    print(f"  3. Melhor modelo: {best_model} (MAPE: {best_mape:.2f}%)")

    print("\nPróximos passos sugeridos:")
    print("  1. Ajustar hiperparâmetros dos modelos com Optuna")
    print("  2. Testar ensemble de múltiplos modelos")
    print("  3. Integrar com agente RL avançado para otimização dos pesos")
    print("  4. Validar em dados reais de indicadores econômicos")

    print("\n" + "=" * 80)
    print("FIM DO EXEMPLO")
    print("=" * 80)


if __name__ == "__main__":
    main()
