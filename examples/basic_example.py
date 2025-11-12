"""
Exemplo básico de uso do framework de RL para previsão de séries temporais.

Este exemplo demonstra como:
1. Gerar dados sintéticos
2. Criar modelos supervisionados
3. Criar ensemble
4. Treinar agente RL
5. Avaliar resultados
"""

import numpy as np
import pandas as pd
import sys
import os

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_utils import generate_synthetic_data, split_data
from src.models.arima_model import ARIMAPredictor
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.ensemble_predictor import EnsemblePredictor
from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.training.trainer import RLTrainer
from src.utils.visualization import plot_predictions, plot_coefficients
from src.utils.metrics import calculate_metrics


def main():
    print("="*80)
    print("FRAMEWORK DE RL PARA PREVISÃO DE SÉRIES TEMPORAIS ECONÔMICAS")
    print("="*80)

    # 1. Geração de dados
    print("\n[1/6] Gerando dados sintéticos...")
    data = generate_synthetic_data(
        n_points=300,
        trend=0.05,
        seasonality_amplitude=15.0,
        seasonality_period=12,
        noise_std=3.0,
        seed=42
    )
    print(f"✓ Dados gerados: {len(data)} pontos")

    # Divide dados
    train_data, val_data, test_data = split_data(data, train_ratio=0.7, val_ratio=0.15)
    print(f"  Treino: {len(train_data)} | Validação: {len(val_data)} | Teste: {len(test_data)}")

    # 2. Criação de modelos supervisionados
    print("\n[2/6] Criando modelos supervisionados base...")

    models = [
        ARIMAPredictor(order=(2, 1, 2), name="ARIMA"),
        LSTMPredictor(lookback=12, hidden_size=32, num_layers=2, epochs=50, name="LSTM"),
        XGBoostPredictor(lookback=12, n_estimators=50, name="XGBoost")
    ]

    print(f"✓ {len(models)} modelos criados: {[m.name for m in models]}")

    # 3. Criação do ensemble
    print("\n[3/6] Criando ensemble de modelos...")
    ensemble = EnsemblePredictor(models)

    # Treina ensemble com dados de treino
    ensemble.fit(train_data['value'])
    print(f"✓ Ensemble criado e treinado")

    # 4. Criação do ambiente de RL
    print("\n[4/6] Criando ambiente de RL...")
    env = TimeSeriesEnv(
        data=train_data,
        forecast_horizon=6,  # Previsão de 6 meses
        window_size=24,
        n_coefficients=len(models),
        max_steps=50
    )
    print(f"✓ Ambiente criado:")
    print(f"  - Horizonte de previsão: {env.forecast_horizon} meses")
    print(f"  - Janela de observação: {env.window_size} pontos")
    print(f"  - Número de coeficientes: {env.n_coefficients}")

    # 5. Criação e treinamento do agente RL
    print("\n[5/6] Criando e treinando agente RL...")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = RLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        hidden_dim=128,
        device='cpu'
    )

    print(f"✓ Agente PPO criado:")
    print(f"  - Dimensão do estado: {state_dim}")
    print(f"  - Dimensão da ação: {action_dim}")

    # Trainer
    trainer = RLTrainer(env, agent, ensemble)

    # Treinamento
    print("\n  Iniciando treinamento...")
    history = trainer.train(
        n_episodes=200,
        max_steps=50,
        eval_frequency=50,
        save_frequency=100,
        verbose=True
    )

    # 6. Avaliação
    print("\n[6/6] Avaliando modelo treinado...")

    # Avaliação no agente
    eval_results = trainer.evaluate(n_episodes=10, deterministic=True, verbose=True)

    # Pega melhores coeficientes
    best_coefficients = trainer.get_best_coefficients()

    if best_coefficients is not None:
        print(f"\n✓ Melhores coeficientes encontrados:")
        for i, (model, coef) in enumerate(zip(models, best_coefficients)):
            print(f"  {model.name}: {coef:.4f}")

        # Atualiza ensemble com melhores coeficientes
        ensemble.update_weights(best_coefficients)

    # Testa ensemble otimizado em dados de teste
    print(f"\n{'='*80}")
    print("TESTE NO CONJUNTO DE TESTE")
    print(f"{'='*80}")

    # Retreina ensemble com todos os dados de treino + validação
    full_train_data = pd.concat([train_data, val_data])
    ensemble.fit(full_train_data['value'])

    # Faz previsão
    forecast_horizon = 12
    predictions = ensemble.predict(steps=forecast_horizon)

    # Limita ao tamanho real dos dados de teste
    actual_values = test_data['value'].values[:forecast_horizon]
    predictions = predictions[:len(actual_values)]

    # Calcula métricas
    metrics = calculate_metrics(actual_values, predictions)

    print(f"\nMétricas no conjunto de teste:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Acurácia Direcional: {metrics['directional_accuracy']:.2f}%")

    # Visualizações
    print(f"\n{'='*80}")
    print("VISUALIZAÇÕES")
    print(f"{'='*80}\n")

    # Plot previsões
    plot_predictions(
        actual_values,
        predictions,
        title="Ensemble Otimizado por RL - Teste"
    )

    # Plot coeficientes
    if best_coefficients is not None:
        plot_coefficients(
            best_coefficients,
            model_names=[m.name for m in models],
            title="Coeficientes Otimizados pelo Agente RL"
        )

    # Plot histórico de treinamento
    trainer.plot_training_progress()

    print(f"\n{'='*80}")
    print("✓ EXEMPLO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
