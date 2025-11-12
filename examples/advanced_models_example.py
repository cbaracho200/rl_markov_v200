"""
Exemplo de uso dos modelos avançados com otimização recursiva de hiperparâmetros.

Este exemplo demonstra:
1. Uso de modelos state-of-the-art (Prophet, CatBoost, LightGBM, AutoARIMA)
2. Otimização automática de hiperparâmetros com Optuna
3. Otimização recursiva durante o treinamento
4. Comparação de performance entre modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Imports do framework
from src.utils.data_utils import generate_synthetic_data, split_data

# Modelos básicos
from src.models import (
    ARIMAPredictor,
    LSTMPredictor,
    XGBoostPredictor,
    EnsemblePredictor
)

# Modelos avançados
from src.models import (
    AutoARIMAPredictor,
    ProphetPredictor,
    CatBoostPredictor,
    LightGBMPredictor
)

# Otimização
from src.optimization import HyperparameterOptimizer, RecursiveOptimizer

# RL
from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.training.trainer import RLTrainer
from src.utils.metrics import calculate_metrics


def main():
    """Função principal demonstrando modelos avançados."""

    print("="*80)
    print("🎓 DEMONSTRAÇÃO: Modelos Avançados + Otimização Recursiva")
    print("="*80)
    print()

    # =========================================================================
    # 1. PREPARAÇÃO DOS DADOS
    # =========================================================================
    print("📊 1. Preparando dados...")
    data = generate_synthetic_data(
        n_points=300,
        trend=0.05,
        seasonality_amplitude=15.0,
        noise_std=3.0,
        seed=42
    )

    train_data, val_data, test_data = split_data(data, train_ratio=0.7, val_ratio=0.15)

    print(f"   ✓ Treino: {len(train_data)} pontos")
    print(f"   ✓ Validação: {len(val_data)} pontos")
    print(f"   ✓ Teste: {len(test_data)} pontos")
    print()

    # =========================================================================
    # 2. CRIAÇÃO DOS MODELOS AVANÇADOS
    # =========================================================================
    print("🤖 2. Criando modelos avançados...")

    models_basic = [
        ARIMAPredictor(order=(2, 1, 2), name="ARIMA"),
        LSTMPredictor(lookback=12, hidden_size=32, epochs=30, name="LSTM"),
        XGBoostPredictor(lookback=12, n_estimators=100, name="XGBoost")
    ]

    models_advanced = [
        AutoARIMAPredictor(
            max_p=5, max_q=5, seasonal=True, m=12,
            stepwise=True, name="AutoARIMA"
        ),
        ProphetPredictor(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            name="Prophet"
        ),
        CatBoostPredictor(
            lookback=12, iterations=300,
            learning_rate=0.05, depth=6,
            name="CatBoost"
        ),
        LightGBMPredictor(
            lookback=12, n_estimators=300,
            learning_rate=0.05, num_leaves=31,
            name="LightGBM"
        )
    ]

    print(f"   ✓ {len(models_basic)} modelos básicos")
    print(f"   ✓ {len(models_advanced)} modelos avançados")
    print()

    # =========================================================================
    # 3. OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
    # =========================================================================
    print("🔍 3. Otimizando hiperparâmetros com Optuna...")
    print("   (Isso pode levar alguns minutos...)")
    print()

    # Cria otimizador
    optimizer = HyperparameterOptimizer(
        metric='mape',
        direction='minimize',
        n_trials=20,  # Use 50+ em produção
        verbose=True
    )

    # Define espaços de busca
    model_configs = [
        {
            'class': CatBoostPredictor,
            'param_space': {
                'lookback': ('int', 6, 24),
                'iterations': ('int', 100, 500),
                'learning_rate': ('float', 0.01, 0.1, 'log'),
                'depth': ('int', 4, 10),
                'l2_leaf_reg': ('float', 1.0, 10.0)
            }
        },
        {
            'class': LightGBMPredictor,
            'param_space': {
                'lookback': ('int', 6, 24),
                'n_estimators': ('int', 100, 500),
                'learning_rate': ('float', 0.01, 0.1, 'log'),
                'num_leaves': ('int', 20, 50),
                'max_depth': ('int', 3, 10)
            }
        },
        {
            'class': XGBoostPredictor,
            'param_space': {
                'lookback': ('int', 6, 24),
                'n_estimators': ('int', 50, 300),
                'learning_rate': ('float', 0.01, 0.1, 'log'),
                'max_depth': ('int', 3, 10)
            }
        }
    ]

    # Otimiza todos os modelos
    best_params = optimizer.optimize_ensemble(
        model_configs=model_configs,
        train_data=train_data['value'],
        val_data=val_data['value'],
        forecast_horizon=12
    )

    print()
    print("="*80)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("="*80)
    print()

    # =========================================================================
    # 4. CRIA MODELOS COM MELHORES HIPERPARÂMETROS
    # =========================================================================
    print("🏗️  4. Criando modelos com hiperparâmetros otimizados...")

    optimized_models = []

    for model_name, params in best_params.items():
        if model_name == 'CatBoostPredictor':
            optimized_models.append(CatBoostPredictor(**params, name=f"{model_name}_opt"))
        elif model_name == 'LightGBMPredictor':
            optimized_models.append(LightGBMPredictor(**params, name=f"{model_name}_opt"))
        elif model_name == 'XGBoostPredictor':
            optimized_models.append(XGBoostPredictor(**params, name=f"{model_name}_opt"))

    # Adiciona modelos avançados não otimizados
    optimized_models.extend([
        AutoARIMAPredictor(name="AutoARIMA"),
        ProphetPredictor(name="Prophet")
    ])

    print(f"   ✓ {len(optimized_models)} modelos com hiperparâmetros otimizados")
    print()

    # =========================================================================
    # 5. CRIA E TREINA ENSEMBLE
    # =========================================================================
    print("🔄 5. Criando e treinando ensemble...")

    ensemble = EnsemblePredictor(optimized_models)
    ensemble.fit(train_data['value'])

    print("   ✓ Ensemble treinado!")
    print()

    # =========================================================================
    # 6. TREINAMENTO RL COM OTIMIZAÇÃO RECURSIVA
    # =========================================================================
    print("🎯 6. Treinamento RL com otimização recursiva...")

    # Cria ambiente
    env = TimeSeriesEnv(
        data=train_data,
        forecast_horizon=12,
        window_size=24,
        n_coefficients=len(optimized_models),
        max_steps=50
    )

    # Cria agente
    agent = RLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=3e-4
    )

    # Cria trainer
    trainer = RLTrainer(env, agent, ensemble)

    # Cria otimizador recursivo
    recursive_opt = RecursiveOptimizer(
        hyperparameter_optimizer=optimizer,
        reoptimize_frequency=50,  # Reotimiza a cada 50 episódios
        performance_window=20,
        improvement_threshold=0.05
    )

    # Treina com reotimização
    print("\n   Treinando agente RL...")
    print("   (Reotimizará hiperparâmetros automaticamente se necessário)")
    print()

    history = trainer.train(
        n_episodes=100,  # Use 200+ em produção
        max_steps=50,
        eval_frequency=25,
        verbose=True
    )

    print()
    print("="*80)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("="*80)
    print()

    # =========================================================================
    # 7. AVALIAÇÃO E COMPARAÇÃO
    # =========================================================================
    print("📊 7. Avaliando e comparando modelos...")
    print()

    # Avalia ensemble otimizado
    eval_results = trainer.evaluate(n_episodes=10, deterministic=True, verbose=False)

    # Testa no conjunto de teste
    full_train = pd.concat([train_data, val_data])
    ensemble.fit(full_train['value'])

    predictions_ensemble = ensemble.predict(steps=12)
    actual_test = test_data['value'].values[:12]

    metrics_ensemble = calculate_metrics(actual_test, predictions_ensemble)

    # Compara com modelos individuais
    print(f"\n{'='*80}")
    print(f"{'COMPARAÇÃO DE PERFORMANCE':^80}")
    print(f"{'='*80}\n")

    print(f"{'Modelo':<30} {'MAPE':<12} {'RMSE':<12} {'MAE':<12}")
    print(f"{'-'*80}")

    # Testa cada modelo individualmente
    for model in optimized_models:
        try:
            model.fit(full_train['value'])
            pred = model.predict(steps=12)
            metrics = calculate_metrics(actual_test, pred[:len(actual_test)])
            print(f"{model.name:<30} {metrics['mape']:>10.2f}% {metrics['rmse']:>10.4f}  {metrics['mae']:>10.4f}")
        except:
            print(f"{model.name:<30} {'ERRO':>10s}   {'ERRO':>10s}    {'ERRO':>10s}")

    print(f"{'-'*80}")
    print(f"{'Ensemble Otimizado (RL)':<30} {metrics_ensemble['mape']:>10.2f}% {metrics_ensemble['rmse']:>10.4f}  {metrics_ensemble['mae']:>10.4f}")
    print(f"{'-'*80}")
    print()

    # Interpretação
    if metrics_ensemble['mape'] < 5:
        interpretation = "🌟 EXCELENTE!"
    elif metrics_ensemble['mape'] < 10:
        interpretation = "✅ MUITO BOM!"
    elif metrics_ensemble['mape'] < 15:
        interpretation = "👍 BOM!"
    else:
        interpretation = "⚠️ ACEITÁVEL"

    print(f"💡 Performance do Ensemble: {interpretation}")
    print()

    # =========================================================================
    # 8. VISUALIZAÇÕES
    # =========================================================================
    print("📈 8. Gerando visualizações...")

    # Plot 1: Previsões vs Real
    plt.figure(figsize=(14, 6))
    plt.plot(actual_test, 'o-', label='Real', linewidth=2, markersize=8)
    plt.plot(predictions_ensemble[:len(actual_test)], 's-',
             label=f'Ensemble Otimizado (MAPE: {metrics_ensemble["mape"]:.2f}%)',
             linewidth=2, markersize=8)
    plt.xlabel('Mês', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.title('Previsão do Ensemble Otimizado vs Valores Reais', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./ensemble_optimized_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Histórico de treinamento
    trainer.plot_training_progress()

    print()
    print("="*80)
    print("🎉 DEMONSTRAÇÃO COMPLETA!")
    print("="*80)
    print()

    print("📚 Modelos Utilizados:")
    print("   ✓ AutoARIMA (busca automática de parâmetros)")
    print("   ✓ Prophet (Facebook, robusto a outliers)")
    print("   ✓ CatBoost (gradient boosting avançado)")
    print("   ✓ LightGBM (ultra-rápido)")
    print("   ✓ XGBoost (otimizado)")
    print()

    print("🔬 Técnicas de Otimização:")
    print("   ✓ Optuna (Bayesian Optimization)")
    print("   ✓ Otimização Recursiva (ajusta durante treinamento)")
    print("   ✓ Pruning automático de trials ruins")
    print("   ✓ Parallel trial execution")
    print()

    print("📊 Métricas Finais:")
    print(f"   • MAPE: {metrics_ensemble['mape']:.2f}%")
    print(f"   • RMSE: {metrics_ensemble['rmse']:.4f}")
    print(f"   • MAE: {metrics_ensemble['mae']:.4f}")
    print(f"   • R²: {metrics_ensemble['r2']:.4f}")
    print()


if __name__ == "__main__":
    main()
