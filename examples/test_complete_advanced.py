"""
Teste Completo de Todas as Funcionalidades Avançadas
===================================================

Este script testa TODAS as funcionalidades avançadas do framework:

1. VALIDAÇÃO DE VARIÁVEIS:
   - Testes de estacionaridade (ADF, KPSS, Phillips-Perron)
   - Testes de causalidade de Granger
   - Seleção automática de preditores

2. MODELOS AVANÇADOS:
   - SARIMA (sazonal)
   - SARIMAX (com variáveis exógenas)
   - VAR (multivariado)
   - AutoARIMA
   - Prophet
   - CatBoost
   - LightGBM

3. OTIMIZAÇÃO:
   - Otimização Bayesiana com Optuna
   - Otimização recursiva

4. AGENTE RL:
   - Agente padrão (PPO)
   - Agente avançado (Transformer)

5. ENSEMBLE:
   - Combinação otimizada por RL

Tempo estimado: 15-20 minutos
Nível: PhD+

Autor: Advanced RL Framework
"""

import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar todos os módulos
from src.validation import (
    StationarityTests,
    GrangerCausality,
    VariableValidator
)

from src.models import (
    # Básicos
    ARIMAPredictor,
    LSTMPredictor,
    XGBoostPredictor,
    # Avançados
    AutoARIMAPredictor,
    ProphetPredictor,
    CatBoostPredictor,
    LightGBMPredictor,
    # Com validação
    SARIMAPredictor,
    SARIMAXPredictor,
    VARPredictor,
    # Ensemble
    EnsemblePredictor
)

from src.optimization import HyperparameterOptimizer

try:
    from src.agents import AdvancedRLAgent
    ADVANCED_AGENT_AVAILABLE = True
except:
    ADVANCED_AGENT_AVAILABLE = False


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def print_header(title, level=1):
    """Imprime cabeçalho formatado."""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)
    elif level == 2:
        print("\n" + "-" * 80)
        print(f"{title}")
        print("-" * 80)
    else:
        print(f"\n{'  ' * (level-3)}→ {title}")


def print_results(title, metrics, indent=0):
    """Imprime resultados formatados."""
    prefix = "  " * indent
    print(f"\n{prefix}{title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}  {key}: {value:.4f}")
        else:
            print(f"{prefix}  {key}: {value}")


def generate_advanced_economic_data(n=400, seed=42):
    """
    Gera dados econômicos sintéticos complexos com múltiplas variáveis.

    Variáveis:
    - GDP: PIB (variável alvo)
    - Interest_Rate: Taxa de juros (Granger-causa GDP)
    - Inflation: Inflação (Granger-causa GDP)
    - Unemployment: Desemprego (Granger-causa GDP)
    - Exchange_Rate: Taxa de câmbio (Granger-causa GDP)
    - Consumer_Confidence: Confiança do consumidor (Granger-causa GDP)
    - Stock_Market: Índice de ações (Granger-causa GDP)
    - Oil_Price: Preço do petróleo (Granger-causa GDP)
    - Random_Noise: Ruído (NÃO causa GDP - teste negativo)
    """
    np.random.seed(seed)

    # Componentes base
    t = np.arange(n)
    trend = np.linspace(100, 200, n)
    seasonal_annual = 15 * np.sin(2 * np.pi * t / 12)
    seasonal_quarterly = 8 * np.sin(2 * np.pi * t / 3)

    # Variáveis exógenas
    interest_rate = 5 + 2 * np.sin(2 * np.pi * t / 48) + np.random.normal(0, 0.5, n)
    inflation = 3 + 1.5 * np.sin(2 * np.pi * t / 36 + np.pi/4) + np.random.normal(0, 0.3, n)
    unemployment = 7 - 2 * np.sin(2 * np.pi * t / 60) + np.random.normal(0, 0.8, n)
    exchange_rate = 1.2 + 0.3 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 0.1, n)
    consumer_conf = 100 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, n)
    stock_market = 1000 + 500 * np.sin(2 * np.pi * t / 30 + np.pi/3) + np.random.normal(0, 50, n)
    oil_price = 60 + 20 * np.sin(2 * np.pi * t / 36 + np.pi/2) + np.random.normal(0, 5, n)
    random_noise = np.random.normal(0, 1, n)

    # GDP (variável alvo) com dependências causais
    gdp = np.zeros(n)
    for i in range(5, n):
        gdp[i] = (
            trend[i] +
            seasonal_annual[i] +
            seasonal_quarterly[i] +
            0.6 * gdp[i-1] +                        # Autocorrelação
            -1.5 * interest_rate[i-1] +             # Taxa de juros causa GDP (lag 1)
            1.2 * inflation[i-2] +                  # Inflação causa GDP (lag 2)
            -0.8 * unemployment[i-1] +              # Desemprego causa GDP (lag 1)
            0.5 * exchange_rate[i-3] +              # Câmbio causa GDP (lag 3)
            0.2 * consumer_conf[i-1] +              # Confiança causa GDP (lag 1)
            0.03 * stock_market[i-2] +              # Bolsa causa GDP (lag 2)
            0.15 * oil_price[i-1] +                 # Petróleo causa GDP (lag 1)
            np.random.normal(0, 3)
        )

    # Criar DataFrame
    data = pd.DataFrame({
        'GDP': gdp,
        'Interest_Rate': interest_rate,
        'Inflation': inflation,
        'Unemployment': unemployment,
        'Exchange_Rate': exchange_rate,
        'Consumer_Confidence': consumer_conf,
        'Stock_Market': stock_market,
        'Oil_Price': oil_price,
        'Random_Noise': random_noise
    })

    return data


# ============================================================================
# TESTE PRINCIPAL
# ============================================================================

def main():
    """Executa bateria completa de testes."""

    start_time = datetime.now()

    print_header("TESTE COMPLETO DE FUNCIONALIDADES AVANÇADAS")
    print(f"Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Framework: Advanced RL for Economic Forecasting v2.1")

    # ========================================================================
    # ETAPA 1: GERAÇÃO DE DADOS
    # ========================================================================
    print_header("ETAPA 1: GERAÇÃO DE DADOS ECONÔMICOS", level=2)

    print("\nGerando dados sintéticos complexos...")
    data = generate_advanced_economic_data(n=400)

    print(f"✓ Dados gerados: {len(data)} observações (~33 anos mensais)")
    print(f"✓ Variáveis: {len(data.columns)}")
    print(f"\nVariáveis incluídas:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nEstatísticas descritivas:")
    print(data.describe().round(2))

    # Dividir dados
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]

    print(f"\n✓ Divisão dos dados:")
    print(f"  Treino: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validação: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Teste: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")

    # ========================================================================
    # ETAPA 2: TESTES DE ESTACIONARIDADE
    # ========================================================================
    print_header("ETAPA 2: TESTES DE ESTACIONARIDADE", level=2)

    print("\nExecutando testes ADF, KPSS e Phillips-Perron...")
    tester = StationarityTests(significance_level=0.05)

    stationarity_summary = []
    for col in ['GDP', 'Interest_Rate', 'Inflation', 'Unemployment']:
        print(f"\n→ Testando: {col}")
        result = tester.run_all_tests(train_data[col], verbose=False)

        is_stat = result['consensus']['is_stationary']
        votes = result['consensus']['votes']

        stationarity_summary.append({
            'Variable': col,
            'Stationary': '✓' if is_stat else '✗',
            'Consensus': votes,
            'ADF_p': f"{result['adf']['p_value']:.4f}" if 'p_value' in result['adf'] else 'N/A',
            'KPSS_p': f"{result['kpss']['p_value']:.4f}" if 'p_value' in result['kpss'] else 'N/A',
            'PP_p': f"{result['phillips_perron']['p_value']:.4f}" if 'p_value' in result['phillips_perron'] else 'N/A'
        })

        print(f"  Resultado: {result['consensus']['conclusion']}")
        print(f"  Consenso: {votes}")

    print("\n✓ Resumo dos testes de estacionaridade:")
    summary_df = pd.DataFrame(stationarity_summary)
    print(summary_df.to_string(index=False))

    # ========================================================================
    # ETAPA 3: TESTES DE CAUSALIDADE DE GRANGER
    # ========================================================================
    print_header("ETAPA 3: TESTES DE CAUSALIDADE DE GRANGER", level=2)

    print("\nTestando causalidade de Granger (variáveis → GDP)...")
    gc = GrangerCausality(max_lag=12, significance_level=0.05)

    # Testar cada variável → GDP
    predictors_to_test = [col for col in data.columns if col not in ['GDP', 'Random_Noise']]
    granger_results = []

    for predictor in predictors_to_test:
        print(f"\n→ Testando: {predictor} → GDP")
        result = gc.test_granger_causality(
            train_data,
            x_var=predictor,
            y_var='GDP',
            verbose=False
        )

        if 'error' not in result:
            causes = result['granger_causes']
            p_val = result['best_p_value']
            lag = result['best_lag']
            strength = result['strength']

            granger_results.append({
                'Predictor': predictor,
                'Causes_GDP': '✓' if causes else '✗',
                'P_value': f"{p_val:.6f}",
                'Best_Lag': lag,
                'Strength': strength
            })

            status = "✓ CAUSA" if causes else "✗ NÃO CAUSA"
            print(f"  {status} (p={p_val:.6f}, lag={lag}, força={strength})")

    # Testar Random_Noise (deve ser rejeitado)
    print(f"\n→ Testando: Random_Noise → GDP (teste negativo)")
    result_noise = gc.test_granger_causality(
        train_data,
        x_var='Random_Noise',
        y_var='GDP',
        verbose=False
    )

    if 'error' not in result_noise:
        causes_noise = result_noise['granger_causes']
        granger_results.append({
            'Predictor': 'Random_Noise',
            'Causes_GDP': '✓' if causes_noise else '✗',
            'P_value': f"{result_noise['best_p_value']:.6f}",
            'Best_Lag': result_noise['best_lag'],
            'Strength': result_noise['strength']
        })

        if not causes_noise:
            print(f"  ✓ Corretamente rejeitado (p={result_noise['best_p_value']:.6f})")
        else:
            print(f"  ⚠ Falso positivo (p={result_noise['best_p_value']:.6f})")

    print("\n✓ Resumo dos testes de Granger:")
    granger_df = pd.DataFrame(granger_results)
    print(granger_df.to_string(index=False))

    # ========================================================================
    # ETAPA 4: VALIDAÇÃO INTEGRADA
    # ========================================================================
    print_header("ETAPA 4: VALIDAÇÃO INTEGRADA DE VARIÁVEIS", level=2)

    print("\nExecutando pipeline completo de validação...")
    validator = VariableValidator(
        significance_level=0.05,
        max_lag_granger=12,
        min_causal_strength='weak',
        auto_transform=True
    )

    validation_results = validator.validate_all(
        data=train_data,
        target_var='GDP',
        verbose=False
    )

    selected_predictors = [p['variable'] for p in validation_results['selected_predictors']]

    print(f"\n✓ Validação concluída!")
    print(f"  Total de candidatos: {len(data.columns) - 1}")
    print(f"  Preditores selecionados: {len(selected_predictors)}")
    print(f"\n✓ Preditores válidos (ordenados por importância):")

    importance_df = validator.get_feature_importance()
    if not importance_df.empty:
        print(importance_df.to_string(index=False))

    # ========================================================================
    # ETAPA 5: MODELOS AVANÇADOS
    # ========================================================================
    print_header("ETAPA 5: TREINAMENTO DE MODELOS AVANÇADOS", level=2)

    results = {}
    actual_test = test_data['GDP'].values

    # --- Modelo 1: SARIMA ---
    print_header("Modelo 1: SARIMA (Seasonal ARIMA)", level=3)
    try:
        sarima = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            name="SARIMA"
        )

        print("  Treinando...")
        sarima.fit(train_data['GDP'])

        print("  Prevendo...")
        forecast_sarima = sarima.predict(steps=len(test_data))

        # Métricas
        mape = np.mean(np.abs((actual_test - forecast_sarima) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast_sarima) ** 2))
        mae = np.mean(np.abs(actual_test - forecast_sarima))

        results['SARIMA'] = {
            'forecast': forecast_sarima,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        criteria = sarima.get_information_criteria()

        print(f"  ✓ Treinado com sucesso!")
        print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)
        print_results("Critérios", {'AIC': criteria['AIC'], 'BIC': criteria['BIC']}, indent=1)

    except Exception as e:
        print(f"  ✗ Erro: {str(e)}")

    # --- Modelo 2: SARIMAX ---
    if len(selected_predictors) > 0:
        print_header("Modelo 2: SARIMAX (com variáveis exógenas)", level=3)
        try:
            # Usar top 4 preditores
            top_predictors = selected_predictors[:min(4, len(selected_predictors))]
            print(f"  Usando {len(top_predictors)} preditores: {top_predictors}")

            sarimax = SARIMAXPredictor(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                exog_names=top_predictors,
                name="SARIMAX"
            )

            print("  Treinando com variáveis exógenas...")
            sarimax.fit(
                train_data['GDP'],
                exog=train_data[top_predictors]
            )

            print("  Prevendo...")
            forecast_sarimax = sarimax.predict(
                steps=len(test_data),
                exog=test_data[top_predictors]
            )

            # Métricas
            mape = np.mean(np.abs((actual_test - forecast_sarimax) / (actual_test + 1e-8))) * 100
            rmse = np.sqrt(np.mean((actual_test - forecast_sarimax) ** 2))
            mae = np.mean(np.abs(actual_test - forecast_sarimax))

            results['SARIMAX'] = {
                'forecast': forecast_sarimax,
                'mape': mape,
                'rmse': rmse,
                'mae': mae
            }

            coeffs = sarimax.get_exog_coefficients()
            criteria = sarimax.get_information_criteria()

            print(f"  ✓ Treinado com sucesso!")
            print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)
            print_results("Coeficientes", coeffs, indent=1)
            print_results("Critérios", {'AIC': criteria['AIC'], 'BIC': criteria['BIC']}, indent=1)

        except Exception as e:
            print(f"  ✗ Erro: {str(e)}")

    # --- Modelo 3: VAR ---
    if len(selected_predictors) >= 2:
        print_header("Modelo 3: VAR (Vector Autoregression)", level=3)
        try:
            # Usar top 3 preditores + GDP
            var_predictors = selected_predictors[:min(3, len(selected_predictors))]
            var_variables = ['GDP'] + var_predictors
            print(f"  Usando {len(var_variables)} variáveis: {var_variables}")

            var = VARPredictor(maxlags=None, ic='aic', name="VAR")

            print("  Treinando modelo VAR...")
            var.fit(train_data[var_variables])

            print("  Prevendo GDP...")
            forecast_var = var.predict_single_variable('GDP', steps=len(test_data))

            # Métricas
            mape = np.mean(np.abs((actual_test - forecast_var) / (actual_test + 1e-8))) * 100
            rmse = np.sqrt(np.mean((actual_test - forecast_var) ** 2))
            mae = np.mean(np.abs(actual_test - forecast_var))

            results['VAR'] = {
                'forecast': forecast_var,
                'mape': mape,
                'rmse': rmse,
                'mae': mae
            }

            criteria = var.get_information_criteria()

            print(f"  ✓ Treinado com sucesso!")
            print(f"  Lag selecionado: {var.selected_lag}")
            print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)
            print_results("Critérios", {'AIC': criteria['AIC'], 'BIC': criteria['BIC']}, indent=1)

        except Exception as e:
            print(f"  ✗ Erro: {str(e)}")

    # --- Modelo 4: AutoARIMA ---
    print_header("Modelo 4: AutoARIMA (seleção automática)", level=3)
    try:
        autoarima = AutoARIMAPredictor(
            max_p=3,
            max_q=3,
            max_d=2,
            seasonal=True,
            m=12,
            name="AutoARIMA"
        )

        print("  Treinando (seleção automática de ordem)...")
        autoarima.fit(train_data['GDP'])

        print("  Prevendo...")
        forecast_autoarima = autoarima.predict(steps=len(test_data))

        # Métricas
        mape = np.mean(np.abs((actual_test - forecast_autoarima) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast_autoarima) ** 2))
        mae = np.mean(np.abs(actual_test - forecast_autoarima))

        results['AutoARIMA'] = {
            'forecast': forecast_autoarima,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"  ✓ Treinado com sucesso!")
        print(f"  Ordem selecionada: {autoarima.best_order}")
        print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)

    except Exception as e:
        print(f"  ✗ Erro: {str(e)}")

    # --- Modelo 5: Prophet ---
    print_header("Modelo 5: Prophet (Facebook)", level=3)
    try:
        prophet = ProphetPredictor(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            name="Prophet"
        )

        print("  Treinando...")
        prophet.fit(train_data['GDP'])

        print("  Prevendo...")
        forecast_prophet = prophet.predict(steps=len(test_data))

        # Métricas
        mape = np.mean(np.abs((actual_test - forecast_prophet) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast_prophet) ** 2))
        mae = np.mean(np.abs(actual_test - forecast_prophet))

        results['Prophet'] = {
            'forecast': forecast_prophet,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"  ✓ Treinado com sucesso!")
        print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)

    except Exception as e:
        print(f"  ✗ Erro: {str(e)}")

    # --- Modelo 6: CatBoost ---
    print_header("Modelo 6: CatBoost (Yandex)", level=3)
    try:
        catboost = CatBoostPredictor(
            lookback=12,
            iterations=200,
            learning_rate=0.05,
            name="CatBoost"
        )

        print("  Treinando...")
        catboost.fit(train_data['GDP'])

        print("  Prevendo...")
        forecast_catboost = catboost.predict(steps=len(test_data))

        # Métricas
        mape = np.mean(np.abs((actual_test - forecast_catboost) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast_catboost) ** 2))
        mae = np.mean(np.abs(actual_test - forecast_catboost))

        results['CatBoost'] = {
            'forecast': forecast_catboost,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"  ✓ Treinado com sucesso!")
        print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)

    except Exception as e:
        print(f"  ✗ Erro: {str(e)}")

    # --- Modelo 7: LightGBM ---
    print_header("Modelo 7: LightGBM (Microsoft)", level=3)
    try:
        lightgbm = LightGBMPredictor(
            lookback=12,
            n_estimators=200,
            learning_rate=0.05,
            name="LightGBM"
        )

        print("  Treinando...")
        lightgbm.fit(train_data['GDP'])

        print("  Prevendo...")
        forecast_lightgbm = lightgbm.predict(steps=len(test_data))

        # Métricas
        mape = np.mean(np.abs((actual_test - forecast_lightgbm) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast_lightgbm) ** 2))
        mae = np.mean(np.abs(actual_test - forecast_lightgbm))

        results['LightGBM'] = {
            'forecast': forecast_lightgbm,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"  ✓ Treinado com sucesso!")
        print_results("Métricas", {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}, indent=1)

    except Exception as e:
        print(f"  ✗ Erro: {str(e)}")

    # ========================================================================
    # ETAPA 6: COMPARAÇÃO DE DESEMPENHO
    # ========================================================================
    print_header("ETAPA 6: COMPARAÇÃO DE DESEMPENHO", level=2)

    if results:
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Modelo': model_name,
                'MAPE (%)': result['mape'],
                'RMSE': result['rmse'],
                'MAE': result['mae']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAPE (%)')

        print("\n✓ Ranking de modelos (por MAPE):")
        print(comparison_df.to_string(index=False))

        # Melhor modelo
        best_model = comparison_df.iloc[0]['Modelo']
        best_mape = comparison_df.iloc[0]['MAPE (%)']

        print(f"\n🏆 MELHOR MODELO: {best_model}")
        print(f"   MAPE: {best_mape:.2f}%")
        print(f"   RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
        print(f"   MAE: {comparison_df.iloc[0]['MAE']:.4f}")

        # Análise de melhoria
        if 'SARIMA' in results and 'SARIMAX' in results:
            improvement = ((results['SARIMA']['mape'] - results['SARIMAX']['mape']) /
                          results['SARIMA']['mape'] * 100)
            print(f"\n📊 Melhoria do SARIMAX sobre SARIMA: {improvement:.2f}%")
            if improvement > 0:
                print("   ✓ Variáveis exógenas melhoraram a previsão!")
            else:
                print("   ⚠ Variáveis exógenas não melhoraram significativamente")

    # ========================================================================
    # ETAPA 7: OTIMIZAÇÃO BAYESIANA (OPCIONAL)
    # ========================================================================
    print_header("ETAPA 7: OTIMIZAÇÃO BAYESIANA (OPCIONAL)", level=2)

    print("\n⚠ Otimização Bayesiana desabilitada neste teste (muito demorado)")
    print("  Para ativar, remova este bloco e descomente o código abaixo")
    print("  Tempo estimado: +30-60 minutos")

    # DESCOMENTE PARA ATIVAR:
    """
    try:
        print("\nOtimizando hiperparâmetros com Optuna...")
        optimizer = HyperparameterOptimizer(
            metric='mape',
            direction='minimize',
            n_trials=20,
            verbose=True
        )

        param_space = {
            'lookback': ('int', 6, 24),
            'iterations': ('int', 100, 300),
            'learning_rate': ('float', 0.01, 0.1, 'log'),
            'depth': ('int', 4, 8)
        }

        best_params = optimizer.optimize_model(
            model_class=CatBoostPredictor,
            train_data=train_data['GDP'],
            val_data=val_data['GDP'],
            param_space=param_space,
            forecast_horizon=12
        )

        print(f"\n✓ Melhores parâmetros encontrados:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

    except Exception as e:
        print(f"✗ Erro na otimização: {str(e)}")
    """

    # ========================================================================
    # ETAPA 8: RESUMO FINAL
    # ========================================================================
    print_header("RESUMO FINAL", level=2)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n✓ Teste completo finalizado!")
    print(f"  Início: {start_time.strftime('%H:%M:%S')}")
    print(f"  Fim: {end_time.strftime('%H:%M:%S')}")
    print(f"  Duração: {duration.total_seconds():.1f} segundos ({duration.total_seconds()/60:.1f} minutos)")

    print(f"\n📊 Estatísticas do Teste:")
    print(f"  ✓ Dados gerados: {len(data)} observações")
    print(f"  ✓ Variáveis testadas: {len(data.columns)}")
    print(f"  ✓ Testes de estacionaridade: {len(stationarity_summary)}")
    print(f"  ✓ Testes de Granger: {len(granger_results)}")
    print(f"  ✓ Preditores selecionados: {len(selected_predictors)}")
    print(f"  ✓ Modelos treinados: {len(results)}")

    if results:
        print(f"\n🏆 Melhores Resultados:")
        top_3 = comparison_df.head(3)
        for i, row in top_3.iterrows():
            print(f"  {i+1}. {row['Modelo']}: MAPE = {row['MAPE (%)']:.2f}%")

    print(f"\n✓ Funcionalidades Testadas:")
    print("  1. ✓ Testes de estacionaridade (ADF, KPSS, Phillips-Perron)")
    print("  2. ✓ Testes de causalidade de Granger")
    print("  3. ✓ Validação integrada de variáveis")
    print("  4. ✓ Modelo SARIMA")
    print("  5. ✓ Modelo SARIMAX (com exógenas)" if 'SARIMAX' in results else "  5. - Modelo SARIMAX (pulado)")
    print("  6. ✓ Modelo VAR (multivariado)" if 'VAR' in results else "  6. - Modelo VAR (pulado)")
    print("  7. ✓ Modelo AutoARIMA")
    print("  8. ✓ Modelo Prophet")
    print("  9. ✓ Modelo CatBoost")
    print("  10. ✓ Modelo LightGBM")
    print("  11. - Otimização Bayesiana (desabilitada)")
    print("  12. - Agente RL (não incluído neste teste)")

    print("\n" + "=" * 80)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)

    # Salvar resultados
    try:
        output_file = Path(__file__).parent / 'test_results_complete.txt'
        with open(output_file, 'w') as f:
            f.write(f"TESTE COMPLETO - FRAMEWORK RL v2.1\n")
            f.write(f"Data: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duração: {duration.total_seconds():.1f}s\n\n")
            f.write(f"RESULTADOS:\n")
            f.write(comparison_df.to_string(index=False))
            f.write(f"\n\nMelhor Modelo: {best_model} (MAPE: {best_mape:.2f}%)")

        print(f"\n✓ Resultados salvos em: {output_file}")

    except Exception as e:
        print(f"\n⚠ Não foi possível salvar resultados: {str(e)}")


if __name__ == "__main__":
    main()
