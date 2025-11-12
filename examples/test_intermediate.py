"""
EXEMPLO INTERMEDIÁRIO: Modelos Avançados + Ensemble + RL

Este exemplo demonstra:
1. Uso dos 4 novos modelos avançados
2. Criação de ensemble
3. Treinamento com agente RL padrão
4. Avaliação e comparação de resultados

Tempo estimado: 5-10 minutos
Nível: Intermediário
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Imports do framework
from src.utils.data_utils import generate_synthetic_data, split_data
from src.models import (
    AutoARIMAPredictor,
    ProphetPredictor,
    CatBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor
)
from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.training.trainer import RLTrainer
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_predictions, plot_coefficients


def print_header(text):
    """Imprime cabeçalho formatado."""
    print(f"\n{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}\n")


def print_section(number, text):
    """Imprime seção formatada."""
    print(f"\n{'─'*80}")
    print(f"📌 {number}. {text}")
    print(f"{'─'*80}\n")


def main():
    """Função principal do exemplo intermediário."""

    print_header("🎓 EXEMPLO INTERMEDIÁRIO: Modelos Avançados + RL")

    # =========================================================================
    # 1. GERAÇÃO DE DADOS
    # =========================================================================
    print_section(1, "Geração de Dados Sintéticos")

    print("Gerando série temporal com:")
    print("  • Tendência crescente")
    print("  • Sazonalidade de 12 meses")
    print("  • Ruído gaussiano")

    data = generate_synthetic_data(
        n_points=250,
        trend=0.08,
        seasonality_amplitude=20.0,
        seasonality_period=12,
        noise_std=4.0,
        seed=42
    )

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.7,
        val_ratio=0.15
    )

    print(f"\n✓ Dados gerados: {len(data)} pontos")
    print(f"  • Treino: {len(train_data)} pontos")
    print(f"  • Validação: {len(val_data)} pontos")
    print(f"  • Teste: {len(test_data)} pontos")

    # Visualiza dados
    plt.figure(figsize=(14, 6))
    plt.plot(train_data['value'], label='Treino', alpha=0.8)
    plt.plot(range(len(train_data), len(train_data) + len(val_data)),
             val_data['value'], label='Validação', alpha=0.8)
    plt.plot(range(len(train_data) + len(val_data), len(data)),
             test_data['value'], label='Teste', alpha=0.8)
    plt.xlabel('Tempo')
    plt.ylabel('Valor')
    plt.title('Série Temporal - Divisão dos Dados', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 2. CRIAÇÃO DOS MODELOS AVANÇADOS
    # =========================================================================
    print_section(2, "Criação de Modelos Avançados")

    print("Criando 4 modelos state-of-the-art:")
    print()

    models = [
        AutoARIMAPredictor(
            max_p=5,
            max_q=5,
            seasonal=True,
            m=12,
            stepwise=True,
            trace=False,
            name="AutoARIMA"
        ),
        ProphetPredictor(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            name="Prophet"
        ),
        CatBoostPredictor(
            lookback=12,
            iterations=200,
            learning_rate=0.05,
            depth=6,
            name="CatBoost"
        ),
        LightGBMPredictor(
            lookback=12,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            name="LightGBM"
        )
    ]

    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name:15s} - Pronto!")

    print(f"\n✓ {len(models)} modelos criados com sucesso!")

    # =========================================================================
    # 3. TREINAMENTO INDIVIDUAL DOS MODELOS
    # =========================================================================
    print_section(3, "Treinamento Individual dos Modelos")

    print("Treinando cada modelo nos dados de treino...")
    print("(Isso pode levar 1-2 minutos)\n")

    trained_models = []
    individual_metrics = {}

    for model in models:
        print(f"[{model.name}] Treinando...", end=" ", flush=True)

        try:
            # Treina modelo
            model.fit(train_data['value'])

            # Testa no conjunto de validação
            predictions = model.predict(steps=len(val_data))
            actual = val_data['value'].values[:len(predictions)]

            # Calcula métricas
            metrics = calculate_metrics(actual, predictions[:len(actual)])
            individual_metrics[model.name] = metrics

            print(f"✓ MAPE: {metrics['mape']:.2f}%")

            trained_models.append(model)

        except Exception as e:
            print(f"✗ Erro: {str(e)[:50]}")

    print(f"\n✓ {len(trained_models)} modelos treinados com sucesso!")

    # =========================================================================
    # 4. CRIAÇÃO DO ENSEMBLE
    # =========================================================================
    print_section(4, "Criação do Ensemble")

    print("Criando ensemble com pesos iguais iniciais...")

    ensemble = EnsemblePredictor(trained_models)

    # Pesos iniciais (iguais)
    initial_weights = ensemble.get_weights()
    print(f"\nPesos iniciais: {initial_weights}")
    print(f"  • Cada modelo: {initial_weights[0]:.3f} ({initial_weights[0]*100:.1f}%)")

    print("\n✓ Ensemble criado!")

    # =========================================================================
    # 5. AMBIENTE E AGENTE RL
    # =========================================================================
    print_section(5, "Configuração do Ambiente de RL")

    FORECAST_HORIZON = 12
    WINDOW_SIZE = 24

    env = TimeSeriesEnv(
        data=train_data,
        forecast_horizon=FORECAST_HORIZON,
        window_size=WINDOW_SIZE,
        n_coefficients=len(trained_models),
        max_steps=50
    )

    print(f"Ambiente criado:")
    print(f"  • Horizonte de previsão: {FORECAST_HORIZON} meses")
    print(f"  • Janela de observação: {WINDOW_SIZE} pontos")
    print(f"  • Número de coeficientes: {len(trained_models)}")
    print(f"  • Dimensão do estado: {env.observation_space.shape[0]}")
    print(f"  • Dimensão da ação: {env.action_space.shape[0]}")

    print("\nCriando agente RL (PPO)...")

    agent = RLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=3e-4,
        gamma=0.99,
        hidden_dim=128
    )

    print(f"✓ Agente PPO criado!")
    print(f"  • Algoritmo: Proximal Policy Optimization")
    print(f"  • Camadas ocultas: 128 neurônios")

    # =========================================================================
    # 6. TREINAMENTO DO AGENTE RL
    # =========================================================================
    print_section(6, "Treinamento do Agente RL")

    print("⏰ Tempo estimado: 3-5 minutos")
    print()

    trainer = RLTrainer(env, agent, ensemble)

    history = trainer.train(
        n_episodes=150,
        max_steps=50,
        eval_frequency=25,
        save_frequency=50,
        verbose=True
    )

    print_header("✅ TREINAMENTO CONCLUÍDO!")

    # =========================================================================
    # 7. EXTRAÇÃO DOS MELHORES COEFICIENTES
    # =========================================================================
    print_section(7, "Melhores Coeficientes Encontrados")

    best_coefficients = trainer.get_best_coefficients()

    if best_coefficients is not None:
        print("🏆 Coeficientes Otimizados pelo RL:\n")

        for model, coef_init, coef_opt in zip(trained_models, initial_weights, best_coefficients):
            change = ((coef_opt - coef_init) / coef_init) * 100
            print(f"  {model.name:15s}: {coef_init:.3f} → {coef_opt:.3f} ({change:+.1f}%)")

        # Atualiza ensemble
        ensemble.update_weights(best_coefficients)
        print("\n✓ Ensemble atualizado com coeficientes otimizados!")

        # Visualiza coeficientes
        plot_coefficients(
            best_coefficients,
            model_names=[m.name for m in trained_models],
            title="Coeficientes Otimizados pelo Agente RL"
        )
    else:
        print("⚠️  Coeficientes não disponíveis")

    # =========================================================================
    # 8. AVALIAÇÃO NO CONJUNTO DE TESTE
    # =========================================================================
    print_section(8, "Avaliação no Conjunto de Teste")

    print("Retreinando ensemble com treino + validação...")

    full_train = pd.concat([train_data, val_data])
    ensemble.fit(full_train['value'])

    print("Fazendo previsões para 12 meses à frente...\n")

    # Previsão do ensemble otimizado
    predictions_ensemble = ensemble.predict(steps=12)
    actual_test = test_data['value'].values[:12]

    # Métricas
    metrics_ensemble = calculate_metrics(actual_test, predictions_ensemble[:len(actual_test)])

    # Comparação com modelos individuais
    print(f"{'─'*80}")
    print(f"{'RESULTADOS FINAIS':^80}")
    print(f"{'─'*80}\n")

    print(f"{'Modelo':<25} {'MAPE':<15} {'RMSE':<15} {'MAE':<15}")
    print(f"{'─'*80}")

    # Modelos individuais
    for model in trained_models:
        try:
            model.fit(full_train['value'])
            pred = model.predict(steps=12)
            metrics = calculate_metrics(actual_test, pred[:len(actual_test)])
            print(f"{model.name:<25} {metrics['mape']:>12.2f}%  {metrics['rmse']:>12.4f}  {metrics['mae']:>12.4f}")
        except:
            print(f"{model.name:<25} {'ERRO':>12s}    {'ERRO':>12s}    {'ERRO':>12s}")

    print(f"{'─'*80}")

    # Ensemble com pesos iguais
    ensemble_equal = EnsemblePredictor(trained_models, weights=np.ones(len(trained_models)))
    ensemble_equal.fit(full_train['value'])
    pred_equal = ensemble_equal.predict(steps=12)
    metrics_equal = calculate_metrics(actual_test, pred_equal[:len(actual_test)])
    print(f"{'Ensemble (pesos iguais)':<25} {metrics_equal['mape']:>12.2f}%  {metrics_equal['rmse']:>12.4f}  {metrics_equal['mae']:>12.4f}")

    # Ensemble otimizado
    print(f"{'Ensemble (otimizado RL)':<25} {metrics_ensemble['mape']:>12.2f}%  {metrics_ensemble['rmse']:>12.4f}  {metrics_ensemble['mae']:>12.4f}")

    print(f"{'─'*80}\n")

    # Melhoria
    improvement = ((metrics_equal['mape'] - metrics_ensemble['mape']) / metrics_equal['mape']) * 100
    print(f"💡 Melhoria do RL: {improvement:.1f}% (MAPE)")

    # Interpretação
    if metrics_ensemble['mape'] < 5:
        interpretation = "🌟 EXCELENTE!"
    elif metrics_ensemble['mape'] < 10:
        interpretation = "✅ MUITO BOM!"
    elif metrics_ensemble['mape'] < 15:
        interpretation = "👍 BOM!"
    else:
        interpretation = "⚠️ ACEITÁVEL"

    print(f"📊 Performance: {interpretation}")

    # =========================================================================
    # 9. VISUALIZAÇÕES FINAIS
    # =========================================================================
    print_section(9, "Visualizações")

    # Plot 1: Previsões vs Real
    plt.figure(figsize=(14, 6))
    plt.plot(actual_test, 'o-', label='Real', linewidth=2, markersize=8, color='black')
    plt.plot(predictions_ensemble[:len(actual_test)], 's-',
             label=f'Ensemble Otimizado (MAPE: {metrics_ensemble["mape"]:.2f}%)',
             linewidth=2, markersize=8, color='red')
    plt.plot(pred_equal[:len(actual_test)], '^-',
             label=f'Ensemble Pesos Iguais (MAPE: {metrics_equal["mape"]:.2f}%)',
             linewidth=2, markersize=6, color='blue', alpha=0.6)
    plt.xlabel('Mês', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.title('Previsões no Conjunto de Teste', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Histórico de treinamento
    trainer.plot_training_progress()

    # =========================================================================
    # 10. RESUMO FINAL
    # =========================================================================
    print_header("📊 RESUMO FINAL")

    print("🎓 Modelos Utilizados:")
    for model in trained_models:
        print(f"  ✓ {model.name}")

    print(f"\n🎯 Resultados:")
    print(f"  • MAPE Ensemble Otimizado: {metrics_ensemble['mape']:.2f}%")
    print(f"  • RMSE: {metrics_ensemble['rmse']:.4f}")
    print(f"  • R²: {metrics_ensemble['r2']:.4f}")
    print(f"  • Melhoria vs Pesos Iguais: {improvement:.1f}%")

    print(f"\n🏆 Melhor Recompensa no Treinamento: {history['best_reward']:.2f}")

    print(f"\n💡 Conclusão:")
    print(f"  O agente RL conseguiu otimizar os coeficientes do ensemble,")
    print(f"  resultando em uma melhoria de {improvement:.1f}% no MAPE!")

    print("\n" + "="*80)
    print("🎉 EXEMPLO INTERMEDIÁRIO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
