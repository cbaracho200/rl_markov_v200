"""
PREVISÃO DE PIB COM FRAMEWORK COMPLETO
======================================

Este script demonstra o USO COMPLETO do framework com:
1. TODOS os 11 modelos disponíveis (ARIMA, AutoARIMA, SARIMA, SARIMAX, VAR,
   Prophet, XGBoost, LSTM, CatBoost, LightGBM, Ensemble)
2. Otimização Bayesiana de hiperparâmetros com Optuna
3. Salvamento de hiperparâmetros otimizados para reutilização
4. Validação estatística completa
5. Visualizações profissionais
6. Comparação de desempenho

Autor: Sistema de RL para Previsão Econômica
Data: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pickle

warnings.filterwarnings('ignore')

# Importa todos os modelos disponíveis
from src.models import (
    ARIMAPredictor,
    AutoARIMAPredictor,
    SARIMAPredictor,
    SARIMAXPredictor,
    VARPredictor,
    ProphetPredictor,
    XGBoostPredictor,
    LSTMPredictor,
    CatBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor
)

# Importa validação e otimização
from src.validation import VariableValidator
from src.optimization import HyperparameterOptimizer


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class Config:
    """Configuração centralizada do experimento."""

    # Variáveis
    TARGET_VAR = 'pib_acum12m'
    EXOG_VARS = None  # Será detectado automaticamente do dataset

    # Divisão dos dados
    TRAIN_RATIO = 0.65
    VAL_RATIO = 0.20
    TEST_RATIO = 0.15

    # Horizonte de previsão
    FORECAST_HORIZON = 12

    # Otimização
    OPTIMIZE_HYPERPARAMS = True
    N_TRIALS_OPTIMIZATION = 30  # Reduzido para velocidade (use 100+ em produção)
    HYPERPARAMS_FILE = 'outputs/optimized_hyperparams.json'

    # Outputs
    OUTPUT_DIR = 'outputs'
    SAVE_MODELS = True

    # Random seed
    RANDOM_SEED = 42


config = Config()


# ============================================================================
# DETECÇÃO AUTOMÁTICA DE VARIÁVEIS
# ============================================================================

def detect_exogenous_variables(data: pd.DataFrame, target_var: str) -> List[str]:
    """
    Detecta automaticamente variáveis exógenas do dataset.

    Usa todas as colunas exceto o target como variáveis exógenas.

    Args:
        data: DataFrame com os dados
        target_var: Nome da variável alvo

    Returns:
        Lista de nomes de variáveis exógenas
    """
    all_columns = data.columns.tolist()

    # Remove target
    exog_vars = [col for col in all_columns if col != target_var]

    print(f"\n✓ Detecção automática de variáveis:")
    print(f"  Total de colunas: {len(all_columns)}")
    print(f"  Target: {target_var}")
    print(f"  Variáveis exógenas detectadas: {len(exog_vars)}")

    if len(exog_vars) > 10:
        print(f"  Primeiras 10: {exog_vars[:10]}")
    else:
        print(f"  Todas: {exog_vars}")

    return exog_vars


# ============================================================================
# GERAÇÃO DE DADOS SINTÉTICOS
# ============================================================================

def generate_synthetic_pib_data(n_obs: int = 300, n_exog: int = 68) -> pd.DataFrame:
    """
    Gera dados sintéticos realistas para PIB e variáveis exógenas.

    IMPORTANTE: Em produção, substitua por dados reais:
    >>> data = pd.read_csv('seus_dados_pib.csv', parse_dates=['data'], index_col='data')

    Args:
        n_obs: Número de observações
        n_exog: Número de variáveis exógenas a gerar

    Returns:
        DataFrame com todas as variáveis
    """
    np.random.seed(config.RANDOM_SEED)

    dates = pd.date_range(end='2024-01-01', periods=n_obs, freq='M')

    # PIB base com tendência + sazonalidade + ciclos
    t = np.arange(n_obs)
    trend = 100 + 0.3 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 12)
    cycle_short = 8 * np.sin(2 * np.pi * t / 40)
    cycle_long = 12 * np.sin(2 * np.pi * t / 100)
    noise = np.random.normal(0, 2, n_obs)

    pib_base = trend + seasonality + cycle_short + cycle_long + noise

    # Cria DataFrame
    data = pd.DataFrame(index=dates)
    data[config.TARGET_VAR] = pib_base

    # Gera variáveis exógenas correlacionadas
    for i in range(n_exog):
        var_name = f'exog_var_{i+1:03d}'

        # Cada variável tem correlação diferente com PIB
        correlation = np.random.uniform(0.3, 0.9) if i < 30 else np.random.uniform(0.1, 0.5)

        # Componente correlacionado + componente independente
        correlated_component = correlation * pib_base
        independent_component = (1 - correlation) * (
            50 + 0.2 * t +
            3 * np.sin(2 * np.pi * t / 12 + i) +
            np.random.normal(0, 5, n_obs)
        )

        data[var_name] = correlated_component + independent_component

    print(f"✓ Dados sintéticos gerados: {n_obs} observações, {len(data.columns)} variáveis")
    print(f"  IMPORTANTE: Substitua por dados reais em produção!")

    return data


# ============================================================================
# DEFINIÇÃO DE ESPAÇOS DE BUSCA PARA OTIMIZAÇÃO
# ============================================================================

def get_param_spaces() -> Dict[str, Dict]:
    """
    Define espaços de busca de hiperparâmetros para cada modelo.

    Returns:
        Dicionário com espaços de busca
    """
    spaces = {
        'ARIMA': {
            'p': ('int', 0, 5),
            'd': ('int', 0, 2),
            'q': ('int', 0, 5)
        },
        'AutoARIMA': {
            'max_p': ('int', 3, 8),
            'max_q': ('int', 3, 8),
            'max_d': ('int', 1, 3),
            'seasonal': ('categorical', [True, False]),
            'm': ('int', 3, 12)
        },
        'SARIMA': {
            'p': ('int', 0, 3),
            'd': ('int', 0, 2),
            'q': ('int', 0, 3),
            'P': ('int', 0, 2),
            'D': ('int', 0, 1),
            'Q': ('int', 0, 2),
            's': ('int', 4, 12)
        },
        'Prophet': {
            'changepoint_prior_scale': ('float', 0.001, 0.5, 'log'),
            'seasonality_prior_scale': ('float', 0.01, 10.0, 'log'),
            'seasonality_mode': ('categorical', ['additive', 'multiplicative']),
            'changepoint_range': ('float', 0.8, 0.95)
        },
        'XGBoost': {
            'n_estimators': ('int', 50, 300),
            'max_depth': ('int', 3, 10),
            'learning_rate': ('float', 0.01, 0.3, 'log'),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'lookback': ('int', 6, 36)
        },
        'LSTM': {
            'lookback': ('int', 6, 36),
            'hidden_size': ('int', 32, 128),
            'num_layers': ('int', 1, 3),
            'dropout': ('float', 0.1, 0.5),
            'learning_rate': ('float', 0.0001, 0.01, 'log'),
            'epochs': ('int', 50, 200)
        },
        'CatBoost': {
            'iterations': ('int', 100, 500),
            'depth': ('int', 4, 10),
            'learning_rate': ('float', 0.01, 0.3, 'log'),
            'l2_leaf_reg': ('float', 1.0, 10.0),
            'lookback': ('int', 6, 36)
        },
        'LightGBM': {
            'n_estimators': ('int', 50, 300),
            'max_depth': ('int', 3, 10),
            'learning_rate': ('float', 0.01, 0.3, 'log'),
            'num_leaves': ('int', 20, 100),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'lookback': ('int', 6, 36)
        }
    }

    return spaces


# ============================================================================
# OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS
# ============================================================================

def optimize_all_models(
    train_data: pd.Series,
    val_data: pd.Series,
    exog_train: pd.DataFrame = None,
    exog_val: pd.DataFrame = None,
    load_if_exists: bool = True
) -> Dict[str, Dict]:
    """
    Otimiza hiperparâmetros de todos os modelos usando Optuna.

    Args:
        train_data: Dados de treino (target)
        val_data: Dados de validação (target)
        exog_train: Variáveis exógenas de treino
        exog_val: Variáveis exógenas de validação
        load_if_exists: Se True, carrega hiperparâmetros salvos

    Returns:
        Dicionário com melhores hiperparâmetros para cada modelo
    """
    print("\n" + "="*80)
    print("ETAPA: OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS (OPTUNA)")
    print("="*80)

    # Verifica se já existe arquivo salvo
    if load_if_exists and os.path.exists(config.HYPERPARAMS_FILE):
        print(f"\n✓ Carregando hiperparâmetros salvos de: {config.HYPERPARAMS_FILE}")
        with open(config.HYPERPARAMS_FILE, 'r') as f:
            return json.load(f)

    # Cria otimizador
    optimizer = HyperparameterOptimizer(
        metric='mape',
        direction='minimize',
        n_trials=config.N_TRIALS_OPTIMIZATION,
        verbose=True
    )

    # Espaços de busca
    param_spaces = get_param_spaces()

    # Modelos para otimizar (sem VAR e SARIMAX por complexidade, sem Ensemble)
    models_to_optimize = {
        'ARIMA': ARIMAPredictor,
        'AutoARIMA': AutoARIMAPredictor,
        'SARIMA': SARIMAPredictor,
        'Prophet': ProphetPredictor,
        'XGBoost': XGBoostPredictor,
        'LSTM': LSTMPredictor,
        'CatBoost': CatBoostPredictor,
        'LightGBM': LightGBMPredictor
    }

    all_best_params = {}

    for model_name, model_class in models_to_optimize.items():
        print(f"\n{'='*80}")
        print(f"Otimizando: {model_name}")
        print(f"{'='*80}")

        try:
            best_params = optimizer.optimize_model(
                model_class=model_class,
                train_data=train_data,
                val_data=val_data,
                param_space=param_spaces[model_name],
                forecast_horizon=config.FORECAST_HORIZON
            )

            all_best_params[model_name] = best_params

        except Exception as e:
            print(f"\n⚠️  Erro ao otimizar {model_name}: {e}")
            print(f"   Usando hiperparâmetros padrão...")
            all_best_params[model_name] = {}

    # Salva hiperparâmetros otimizados
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.HYPERPARAMS_FILE, 'w') as f:
        json.dump(all_best_params, f, indent=2)

    print(f"\n✓ Hiperparâmetros otimizados salvos em: {config.HYPERPARAMS_FILE}")

    return all_best_params


# ============================================================================
# TREINAMENTO DE TODOS OS MODELOS
# ============================================================================

def train_all_models(
    train_data: pd.Series,
    test_data: pd.Series,
    selected_predictors: List[str],
    data: pd.DataFrame,
    optimized_params: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Treina TODOS os 11 modelos disponíveis no framework.

    Args:
        train_data: Dados de treino
        test_data: Dados de teste
        selected_predictors: Preditores selecionados pela validação
        data: DataFrame completo
        optimized_params: Hiperparâmetros otimizados

    Returns:
        Dicionário com resultados de todos os modelos
    """
    print("\n" + "="*80)
    print("ETAPA: TREINAMENTO DE TODOS OS MODELOS (11 MODELOS)")
    print("="*80)

    target_var = config.TARGET_VAR
    results = {}

    # Prepara dados para modelos com exógenas
    train_idx = train_data.index
    test_idx = test_data.index

    exog_train = data.loc[train_idx, selected_predictors[:10]] if selected_predictors else None
    exog_test = data.loc[test_idx, selected_predictors[:10]] if selected_predictors else None

    # Configuração dos modelos
    models_config = []

    # 1. ARIMA (baseline)
    arima_params = optimized_params.get('ARIMA', {})
    models_config.append({
        'name': 'ARIMA',
        'model': ARIMAPredictor(
            order=(
                arima_params.get('p', 1),
                arima_params.get('d', 1),
                arima_params.get('q', 1)
            ),
            name="ARIMA"
        ),
        'use_exog': False
    })

    # 2. AutoARIMA (automático)
    autoarima_params = optimized_params.get('AutoARIMA', {})
    models_config.append({
        'name': 'AutoARIMA',
        'model': AutoARIMAPredictor(
            max_p=autoarima_params.get('max_p', 5),
            max_q=autoarima_params.get('max_q', 5),
            max_d=autoarima_params.get('max_d', 2),
            seasonal=autoarima_params.get('seasonal', True),
            m=autoarima_params.get('m', 12),
            name="AutoARIMA"
        ),
        'use_exog': False
    })

    # 3. SARIMA (sazonal)
    sarima_params = optimized_params.get('SARIMA', {})
    models_config.append({
        'name': 'SARIMA',
        'model': SARIMAPredictor(
            order=(
                sarima_params.get('p', 1),
                sarima_params.get('d', 1),
                sarima_params.get('q', 1)
            ),
            seasonal_order=(
                sarima_params.get('P', 1),
                sarima_params.get('D', 1),
                sarima_params.get('Q', 1),
                sarima_params.get('s', 12)
            ),
            name="SARIMA"
        ),
        'use_exog': False
    })

    # 4. SARIMAX (sazonal com exógenas)
    if selected_predictors:
        models_config.append({
            'name': 'SARIMAX',
            'model': SARIMAXPredictor(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                exog_names=selected_predictors[:10],
                name="SARIMAX"
            ),
            'use_exog': True,
            'special': 'sarimax'
        })

    # 5. VAR (multivariado) - usa target + top 5 preditores
    if selected_predictors and len(selected_predictors) >= 3:
        models_config.append({
            'name': 'VAR',
            'model': VARPredictor(maxlags=12, ic='aic', name="VAR"),
            'use_exog': False,
            'special': 'var',
            'var_columns': [target_var] + selected_predictors[:5]
        })

    # 6. Prophet (Facebook)
    prophet_params = optimized_params.get('Prophet', {})
    models_config.append({
        'name': 'Prophet',
        'model': ProphetPredictor(
            changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
            seasonality_mode=prophet_params.get('seasonality_mode', 'multiplicative'),
            changepoint_range=prophet_params.get('changepoint_range', 0.9),
            name="Prophet"
        ),
        'use_exog': False
    })

    # 7. XGBoost (gradient boosting)
    xgb_params = optimized_params.get('XGBoost', {})
    models_config.append({
        'name': 'XGBoost',
        'model': XGBoostPredictor(
            lookback=xgb_params.get('lookback', 12),
            n_estimators=xgb_params.get('n_estimators', 100),
            max_depth=xgb_params.get('max_depth', 6),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            subsample=xgb_params.get('subsample', 0.8),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            name="XGBoost"
        ),
        'use_exog': False
    })

    # 8. LSTM (deep learning)
    lstm_params = optimized_params.get('LSTM', {})
    models_config.append({
        'name': 'LSTM',
        'model': LSTMPredictor(
            lookback=lstm_params.get('lookback', 12),
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            learning_rate=lstm_params.get('learning_rate', 0.001),
            epochs=lstm_params.get('epochs', 50),  # Reduzido para velocidade
            name="LSTM"
        ),
        'use_exog': False
    })

    # 9. CatBoost
    catboost_params = optimized_params.get('CatBoost', {})
    models_config.append({
        'name': 'CatBoost',
        'model': CatBoostPredictor(
            lookback=catboost_params.get('lookback', 12),
            iterations=catboost_params.get('iterations', 200),
            depth=catboost_params.get('depth', 6),
            learning_rate=catboost_params.get('learning_rate', 0.1),
            l2_leaf_reg=catboost_params.get('l2_leaf_reg', 3.0),
            name="CatBoost"
        ),
        'use_exog': False
    })

    # 10. LightGBM
    lgbm_params = optimized_params.get('LightGBM', {})
    models_config.append({
        'name': 'LightGBM',
        'model': LightGBMPredictor(
            lookback=lgbm_params.get('lookback', 12),
            n_estimators=lgbm_params.get('n_estimators', 100),
            max_depth=lgbm_params.get('max_depth', 6),
            learning_rate=lgbm_params.get('learning_rate', 0.1),
            num_leaves=lgbm_params.get('num_leaves', 31),
            subsample=lgbm_params.get('subsample', 0.8),
            colsample_bytree=lgbm_params.get('colsample_bytree', 0.8),
            name="LightGBM"
        ),
        'use_exog': False
    })

    # Treina cada modelo
    for i, model_config in enumerate(models_config):
        model_name = model_config['name']
        model = model_config['model']

        print(f"\n[{i+1}/{len(models_config)}] Treinando {model_name}...")

        try:
            # Tratamento especial para VAR (multivariado)
            if model_config.get('special') == 'var':
                var_data_train = data.loc[train_idx, model_config['var_columns']]
                var_data_test = data.loc[test_idx, model_config['var_columns']]

                model.fit(var_data_train)
                forecast = model.predict_single_variable(target_var, steps=len(test_data))

            # Tratamento especial para SARIMAX (precisa de exog futuro)
            elif model_config.get('special') == 'sarimax':
                # Simplificação: treina sem exógenas para previsão futura
                # Em produção, você precisaria fornecer valores futuros de exógenas
                model.fit(train_data, exog=None)
                forecast = model.predict(steps=len(test_data), exog=None)

            # Modelos padrão
            else:
                model.fit(train_data)
                forecast = model.predict(steps=len(test_data))

            # Calcula métricas
            actual = test_data.values
            if len(forecast) != len(actual):
                forecast = forecast[:len(actual)]

            mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100
            rmse = np.sqrt(np.mean((actual - forecast) ** 2))
            mae = np.mean(np.abs(actual - forecast))

            # Salva resultados
            results[model_name] = {
                'model': model,
                'forecast': forecast,
                'actual': actual,
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'params': optimized_params.get(model_name, {})
            }

            print(f"  ✓ {model_name}: MAPE = {mape:.2f}%, RMSE = {rmse:.2f}")

        except Exception as e:
            print(f"  ✗ Erro no {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 11. ENSEMBLE (combina todos os modelos treinados)
    print(f"\n[{len(models_config)+1}/{len(models_config)+1}] Criando Ensemble...")

    try:
        # Pega modelos bem sucedidos
        successful_models = [results[name]['model'] for name in results.keys()]

        if len(successful_models) >= 2:
            # Cria ensemble com pesos iguais
            ensemble = EnsemblePredictor(models=successful_models)
            ensemble.fit(train_data)
            forecast_ensemble = ensemble.predict(steps=len(test_data))

            # Métricas
            mape_ens = np.mean(np.abs((actual - forecast_ensemble) / (actual + 1e-8))) * 100
            rmse_ens = np.sqrt(np.mean((actual - forecast_ensemble) ** 2))
            mae_ens = np.mean(np.abs(actual - forecast_ensemble))

            results['Ensemble'] = {
                'model': ensemble,
                'forecast': forecast_ensemble,
                'actual': actual,
                'mape': mape_ens,
                'rmse': rmse_ens,
                'mae': mae_ens,
                'params': {'n_models': len(successful_models)}
            }

            print(f"  ✓ Ensemble ({len(successful_models)} modelos): MAPE = {mape_ens:.2f}%")
        else:
            print(f"  ⚠️  Ensemble não criado (precisa de pelo menos 2 modelos)")

    except Exception as e:
        print(f"  ✗ Erro no Ensemble: {e}")

    print(f"\n✓ Treinamento concluído: {len(results)} modelos treinados com sucesso")

    return results


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================

def plot_model_comparison(results: Dict, output_dir: str):
    """Plota comparação de todos os modelos."""

    print("\nGerando visualizações...")

    # Configuração de estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)

    # 1. Comparação de previsões
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Subplot 1: Todas as previsões
    ax = axes[0]
    actual = list(results.values())[0]['actual']
    x = np.arange(len(actual))

    ax.plot(x, actual, 'k-', linewidth=2, label='Real', marker='o')

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for i, (name, result) in enumerate(results.items()):
        ax.plot(x, result['forecast'], '--', linewidth=1.5,
                label=f"{name} (MAPE={result['mape']:.2f}%)",
                color=colors[i], alpha=0.7)

    ax.set_title('Comparação de Previsões - Todos os Modelos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Período de Teste')
    ax.set_ylabel('PIB')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Ranking de erros (MAPE)
    ax = axes[1]
    names = list(results.keys())
    mapes = [results[name]['mape'] for name in names]

    sorted_idx = np.argsort(mapes)
    names_sorted = [names[i] for i in sorted_idx]
    mapes_sorted = [mapes[i] for i in sorted_idx]

    colors_sorted = ['green' if m == min(mapes_sorted) else 'steelblue' for m in mapes_sorted]
    bars = ax.barh(names_sorted, mapes_sorted, color=colors_sorted)

    for i, (bar, mape) in enumerate(zip(bars, mapes_sorted)):
        ax.text(mape + 0.1, bar.get_y() + bar.get_height()/2,
                f'{mape:.2f}%', va='center', fontsize=9)

    ax.set_title('Ranking de Modelos por MAPE (menor é melhor)', fontsize=14, fontweight='bold')
    ax.set_xlabel('MAPE (%)')
    ax.set_ylabel('Modelo')
    ax.grid(True, alpha=0.3, axis='x')

    # Subplot 3: Ranking de erros (RMSE)
    ax = axes[2]
    rmses = [results[name]['rmse'] for name in names]

    sorted_idx = np.argsort(rmses)
    names_sorted = [names[i] for i in sorted_idx]
    rmses_sorted = [rmses[i] for i in sorted_idx]

    colors_sorted = ['green' if r == min(rmses_sorted) else 'coral' for r in rmses_sorted]
    bars = ax.barh(names_sorted, rmses_sorted, color=colors_sorted)

    for i, (bar, rmse) in enumerate(zip(bars, rmses_sorted)):
        ax.text(rmse + 0.1, bar.get_y() + bar.get_height()/2,
                f'{rmse:.2f}', va='center', fontsize=9)

    ax.set_title('Ranking de Modelos por RMSE (menor é melhor)', fontsize=14, fontweight='bold')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Modelo')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Análise de resíduos do melhor modelo
    best_model_name = min(results.keys(), key=lambda k: results[k]['mape'])
    best_result = results[best_model_name]

    residuals = best_result['actual'] - best_result['forecast']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Resíduos vs tempo
    axes[0, 0].plot(residuals, marker='o', linestyle='-', color='steelblue')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title(f'Resíduos ao Longo do Tempo - {best_model_name}', fontweight='bold')
    axes[0, 0].set_xlabel('Período')
    axes[0, 0].set_ylabel('Resíduo')
    axes[0, 0].grid(True, alpha=0.3)

    # Histograma de resíduos
    axes[0, 1].hist(residuals, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_title('Distribuição dos Resíduos', fontweight='bold')
    axes[0, 1].set_xlabel('Resíduo')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normalidade dos Resíduos)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Autocorrelação de resíduos
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(pd.Series(residuals), ax=axes[1, 1], color='steelblue')
    axes[1, 1].set_title('Autocorrelação dos Resíduos', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_model_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizações salvas em {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal."""

    print("\n" + "="*80)
    print("FRAMEWORK COMPLETO DE PREVISÃO DE PIB")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {config.TARGET_VAR}")
    print(f"Modelos: 11 (ARIMA, AutoARIMA, SARIMA, SARIMAX, VAR, Prophet,")
    print(f"         XGBoost, LSTM, CatBoost, LightGBM, Ensemble)")
    print("="*80)

    # Cria diretório de outputs
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # ETAPA 1: Carregamento de dados
    # ========================================================================
    print("\nETAPA 1: CARREGAMENTO DE DADOS")
    print("-" * 80)

    data = generate_synthetic_pib_data(n_obs=300)
    print(f"  ✓ Shape: {data.shape}")
    print(f"  ✓ Período: {data.index[0].strftime('%Y-%m')} a {data.index[-1].strftime('%Y-%m')}")

    # Detecta variáveis exógenas automaticamente
    config.EXOG_VARS = detect_exogenous_variables(data, config.TARGET_VAR)

    # ========================================================================
    # ETAPA 2: Divisão dos dados
    # ========================================================================
    print("\nETAPA 2: DIVISÃO DOS DADOS")
    print("-" * 80)

    n = len(data)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_data = data[config.TARGET_VAR].iloc[:train_end]
    val_data = data[config.TARGET_VAR].iloc[train_end:val_end]
    test_data = data[config.TARGET_VAR].iloc[val_end:]

    print(f"  ✓ Treino: {len(train_data)} obs ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"  ✓ Validação: {len(val_data)} obs ({config.VAL_RATIO*100:.0f}%)")
    print(f"  ✓ Teste: {len(test_data)} obs ({config.TEST_RATIO*100:.0f}%)")

    # ========================================================================
    # ETAPA 3: Validação estatística
    # ========================================================================
    print("\nETAPA 3: VALIDAÇÃO ESTATÍSTICA")
    print("-" * 80)

    validator = VariableValidator(verbose=False)
    validation_results = validator.validate_all(
        data=data.iloc[:train_end],
        target_var=config.TARGET_VAR,
        predictor_vars=config.EXOG_VARS
    )

    selected_predictors = validation_results.get('valid_predictors', [])
    print(f"  ✓ Preditores validados: {len(selected_predictors)}")
    if selected_predictors:
        print(f"    Top 10: {selected_predictors[:10]}")

    # ========================================================================
    # ETAPA 4: Otimização de hiperparâmetros
    # ========================================================================
    if config.OPTIMIZE_HYPERPARAMS:
        optimized_params = optimize_all_models(
            train_data=train_data,
            val_data=val_data,
            load_if_exists=True
        )
    else:
        optimized_params = {}
        print("\n⚠️  Otimização de hiperparâmetros desabilitada")

    # ========================================================================
    # ETAPA 5: Treinamento de todos os modelos
    # ========================================================================
    results = train_all_models(
        train_data=train_data,
        test_data=test_data,
        selected_predictors=selected_predictors,
        data=data,
        optimized_params=optimized_params
    )

    # ========================================================================
    # ETAPA 6: Comparação e visualização
    # ========================================================================
    print("\nETAPA 6: COMPARAÇÃO E VISUALIZAÇÃO")
    print("-" * 80)

    # Salva resultados em CSV
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Modelo': name,
            'MAPE (%)': result['mape'],
            'RMSE': result['rmse'],
            'MAE': result['mae']
        })

    df_comparison = pd.DataFrame(comparison_data).sort_values('MAPE (%)')
    df_comparison.to_csv(f"{config.OUTPUT_DIR}/model_comparison.csv", index=False)
    print(f"  ✓ Comparação salva em: {config.OUTPUT_DIR}/model_comparison.csv")

    # Gera visualizações
    plot_model_comparison(results, config.OUTPUT_DIR)

    # ========================================================================
    # ETAPA 7: Salva modelos
    # ========================================================================
    if config.SAVE_MODELS:
        print("\nETAPA 7: SALVAMENTO DE MODELOS")
        print("-" * 80)

        best_model_name = min(results.keys(), key=lambda k: results[k]['mape'])
        best_model = results[best_model_name]['model']

        model_path = f"{config.OUTPUT_DIR}/best_model_{best_model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"  ✓ Melhor modelo salvo: {model_path}")
        print(f"  ✓ Modelo: {best_model_name}")
        print(f"  ✓ MAPE: {results[best_model_name]['mape']:.2f}%")

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)
    print(f"\n{'Modelo':<15} {'MAPE (%)':<12} {'RMSE':<12} {'MAE':<12}")
    print("-" * 80)

    for name, result in sorted(results.items(), key=lambda x: x[1]['mape']):
        print(f"{name:<15} {result['mape']:<12.2f} {result['rmse']:<12.2f} {result['mae']:<12.2f}")

    best_model_name = min(results.keys(), key=lambda k: results[k]['mape'])
    print("\n" + "="*80)
    print(f"🏆 MELHOR MODELO: {best_model_name}")
    print(f"   MAPE: {results[best_model_name]['mape']:.2f}%")
    print(f"   RMSE: {results[best_model_name]['rmse']:.2f}")
    print("="*80)

    print(f"\n✓ Todos os resultados salvos em: {config.OUTPUT_DIR}/")
    print(f"✓ Hiperparâmetros otimizados salvos em: {config.HYPERPARAMS_FILE}")
    print("\nPróximos passos:")
    print("  1. Substitua os dados sintéticos por dados reais")
    print("  2. Ajuste N_TRIALS_OPTIMIZATION para 100+ trials")
    print("  3. Reutilize hiperparâmetros otimizados em novas previsões")
    print("  4. Experimente otimizar os pesos do Ensemble com RL")


if __name__ == "__main__":
    main()
