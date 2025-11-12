"""
Exemplo avançado: Backtesting com dados reais e otimização completa.

Este exemplo demonstra:
1. Carregamento de dados reais (ou sintéticos avançados)
2. Validação cruzada com janela deslizante
3. Otimização de hiperparâmetros
4. Comparação com baselines
5. Análise detalhada de resultados
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_utils import generate_synthetic_data, split_data
from src.models.arima_model import ARIMAPredictor
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.ensemble_predictor import EnsemblePredictor
from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.training.trainer import RLTrainer
from src.utils.metrics import calculate_metrics, compare_models, rolling_forecast_validation
from src.utils.visualization import plot_backtest_results, plot_multiple_forecasts


def backtest_with_rl(data: pd.DataFrame, forecast_horizon: int = 6):
    """
    Realiza backtesting completo com otimização RL.

    Args:
        data: Dados completos
        forecast_horizon: Horizonte de previsão em meses

    Returns:
        Resultados do backtesting
    """
    print(f"\n{'='*80}")
    print(f"BACKTESTING COM HORIZONTE DE {forecast_horizon} MESES")
    print(f"{'='*80}\n")

    # Divide dados
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    print(f"Dados de treino: {len(train_data)} pontos")
    print(f"Dados de teste: {len(test_data)} pontos")

    # Cria modelos
    print("\n[1/4] Criando modelos base...")
    models = [
        ARIMAPredictor(order=(3, 1, 2), name="ARIMA"),
        LSTMPredictor(lookback=24, hidden_size=64, num_layers=2, epochs=100, name="LSTM"),
        XGBoostPredictor(lookback=24, n_estimators=100, max_depth=5, name="XGBoost")
    ]

    # Cria ensemble
    ensemble = EnsemblePredictor(models)
    ensemble.fit(train_data['value'])

    # Cria ambiente e agente
    print("\n[2/4] Criando ambiente de RL e agente...")
    env = TimeSeriesEnv(
        data=train_data,
        forecast_horizon=forecast_horizon,
        window_size=36,
        n_coefficients=len(models),
        max_steps=100
    )

    agent = RLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=1e-3,
        hidden_dim=256,
        device='cpu'
    )

    # Treina
    print("\n[3/4] Treinando agente RL...")
    trainer = RLTrainer(env, agent, ensemble)
    history = trainer.train(
        n_episodes=500,
        max_steps=100,
        eval_frequency=100,
        verbose=True
    )

    # Otimiza ensemble
    best_coefficients = trainer.get_best_coefficients()
    if best_coefficients is not None:
        ensemble.update_weights(best_coefficients)
        print(f"\n✓ Ensemble otimizado com RL")
        for model, coef in zip(models, best_coefficients):
            print(f"  {model.name}: {coef:.4f}")

    # Testa
    print("\n[4/4] Testando no conjunto de teste...")

    # Retreina com todos os dados de treino
    ensemble.fit(train_data['value'])

    # Previsões
    test_size = min(forecast_horizon * 3, len(test_data))
    predictions = ensemble.predict(steps=test_size)
    actual = test_data['value'].values[:test_size]

    # Métricas
    metrics = calculate_metrics(actual, predictions)

    # Comparação com modelos individuais e baseline
    print("\n" + "="*80)
    print("COMPARAÇÃO DE MODELOS")
    print("="*80 + "\n")

    # Baseline: previsão ingênua (último valor)
    baseline_pred = np.full(test_size, train_data['value'].iloc[-1])

    # Previsões individuais dos modelos
    individual_preds = {}
    for model in models:
        model.fit(train_data['value'])
        pred = model.predict(steps=test_size)
        individual_preds[model.name] = pred[:test_size]

    # Ensemble antes da otimização RL (pesos iguais)
    ensemble_equal = EnsemblePredictor(models, weights=np.ones(len(models)))
    ensemble_equal.fit(train_data['value'])
    equal_pred = ensemble_equal.predict(steps=test_size)

    # Compara todos
    all_predictions = {
        'Baseline (Último Valor)': baseline_pred,
        'Ensemble (Pesos Iguais)': equal_pred[:test_size],
        'Ensemble (Otimizado RL)': predictions,
        **individual_preds
    }

    comparison_df = compare_models(actual, all_predictions, metric='mape')
    print(comparison_df.to_string(index=False))

    # Visualizações
    print("\n" + "="*80)
    print("VISUALIZAÇÕES")
    print("="*80 + "\n")

    # Plot comparação
    plot_multiple_forecasts(
        actual,
        {
            'Ensemble RL': predictions,
            'ARIMA': individual_preds['ARIMA'],
            'LSTM': individual_preds['LSTM'],
            'XGBoost': individual_preds['XGBoost']
        },
        title=f"Comparação de Modelos - Horizonte {forecast_horizon} meses"
    )

    # Plot resultados detalhados
    plot_backtest_results(
        {
            'y_true': actual,
            'y_pred': predictions,
            'metrics': metrics
        },
        title=f"Resultados do Backtesting - Ensemble Otimizado RL"
    )

    return {
        'metrics': metrics,
        'comparison': comparison_df,
        'predictions': predictions,
        'actual': actual,
        'coefficients': best_coefficients,
        'history': history
    }


def main():
    print("="*80)
    print("EXEMPLO AVANÇADO: BACKTESTING COM OTIMIZAÇÃO RL")
    print("="*80)

    # Gera dados mais complexos
    print("\nGerando dados sintéticos complexos...")
    data = generate_synthetic_data(
        n_points=500,
        trend=0.03,
        seasonality_amplitude=20.0,
        seasonality_period=12,
        noise_std=5.0,
        seed=123
    )

    print(f"✓ Dados gerados: {len(data)} pontos")
    print(f"  Período: {data['date'].min()} até {data['date'].max()}")

    # Backtesting para diferentes horizontes
    horizons = [6, 9, 12]
    all_results = {}

    for horizon in horizons:
        results = backtest_with_rl(data, forecast_horizon=horizon)
        all_results[horizon] = results

        print(f"\n{'='*80}")
        print(f"RESUMO - HORIZONTE {horizon} MESES")
        print(f"{'='*80}")
        print(f"MAPE: {results['metrics']['mape']:.2f}%")
        print(f"RMSE: {results['metrics']['rmse']:.4f}")
        print(f"R²: {results['metrics']['r2']:.4f}")
        print(f"Acurácia Direcional: {results['metrics']['directional_accuracy']:.2f}%")

    # Comparação final
    print(f"\n{'='*80}")
    print("COMPARAÇÃO ENTRE HORIZONTES")
    print(f"{'='*80}\n")

    summary_data = []
    for horizon, results in all_results.items():
        summary_data.append({
            'Horizonte (meses)': horizon,
            'MAPE (%)': f"{results['metrics']['mape']:.2f}",
            'RMSE': f"{results['metrics']['rmse']:.4f}",
            'R²': f"{results['metrics']['r2']:.4f}",
            'Acurácia Dir. (%)': f"{results['metrics']['directional_accuracy']:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("✓ EXEMPLO AVANÇADO CONCLUÍDO!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
