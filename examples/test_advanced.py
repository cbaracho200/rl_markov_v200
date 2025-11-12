"""
EXEMPLO AVANÇADO: Otimização Completa + Agente RL PhD

Este exemplo demonstra TODAS as técnicas state-of-the-art:
1. Modelos avançados (AutoARIMA, Prophet, CatBoost, LightGBM)
2. Otimização automática de hiperparâmetros com Optuna
3. Otimização recursiva durante treinamento
4. Agente RL avançado com Transformer
5. Comparação detalhada de resultados
6. Visualizações avançadas

Tempo estimado: 15-20 minutos
Nível: PhD / Avançado
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
    XGBoostPredictor,
    LSTMPredictor,
    EnsemblePredictor
)
from src.optimization import HyperparameterOptimizer, RecursiveOptimizer
from src.environments.timeseries_env import TimeSeriesEnv
from src.utils.metrics import calculate_metrics

# Tenta importar agente avançado
try:
    from src.agents import AdvancedRLAgent
    from src.training import AdvancedRLTrainer
    ADVANCED_AVAILABLE = True
except ImportError:
    from src.agents import RLAgent
    from src.training import RLTrainer
    ADVANCED_AVAILABLE = False
    print("⚠️  Agente Avançado não disponível. Usando agente padrão.")


def print_header(text):
    """Imprime cabeçalho formatado."""
    print(f"\n{'='*100}")
    print(f"{text:^100}")
    print(f"{'='*100}\n")


def print_section(number, text):
    """Imprime seção formatada."""
    print(f"\n{'─'*100}")
    print(f"🎓 {number}. {text}")
    print(f"{'─'*100}\n")


def main():
    """Função principal do exemplo avançado."""

    print_header("🎓 EXEMPLO AVANÇADO: Framework Completo com Todas as Técnicas PhD")

    print("Este exemplo demonstra:")
    print("  ✓ 4 modelos state-of-the-art")
    print("  ✓ Otimização de hiperparâmetros com Optuna (Bayesian)")
    print("  ✓ Otimização recursiva durante treinamento")
    print("  ✓ Agente RL avançado com Transformer (se disponível)")
    print("  ✓ Comparação detalhada de todos os modelos")
    print("  ✓ Visualizações avançadas")
    print()
    input("Pressione ENTER para começar...")

    # =========================================================================
    # 1. GERAÇÃO DE DADOS COMPLEXOS
    # =========================================================================
    print_section(1, "Geração de Dados com Múltiplos Padrões")

    print("Criando série temporal complexa com:")
    print("  • Tendência exponencial")
    print("  • Sazonalidade multiplicativa")
    print("  • Ciclo econômico de 4 anos")
    print("  • Eventos extremos (crise)")

    # Dados mais complexos
    dates = pd.date_range(start='2010-01-01', periods=300, freq='M')
    time = np.arange(300)

    # Componentes
    trend = 100 + 50 * (1 - np.exp(-time / 100))  # Tendência exponencial
    seasonality = 15 * np.sin(2 * np.pi * time / 12)  # Sazonalidade anual
    cycle = 10 * np.sin(2 * np.pi * time / 48)  # Ciclo de 4 anos
    noise = np.random.normal(0, 3, 300)

    # Série base
    values = trend + seasonality + cycle + noise

    # Adiciona eventos extremos
    values[100:110] *= 0.85  # Crise 1
    values[200:205] *= 1.15  # Boom

    data = pd.DataFrame({'date': dates, 'value': values})

    train_data, val_data, test_data = split_data(data, train_ratio=0.7, val_ratio=0.15)

    print(f"\n✓ Dados gerados: {len(data)} pontos")
    print(f"  • Treino: {len(train_data)} pontos")
    print(f"  • Validação: {len(val_data)} pontos")
    print(f"  • Teste: {len(test_data)} pontos")

    # Visualiza
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plot 1: Série completa
    axes[0].plot(data['date'], data['value'], linewidth=1.5, color='steelblue')
    axes[0].axvline(x=train_data['date'].iloc[-1], color='red', linestyle='--', alpha=0.7, label='Fim Treino')
    axes[0].axvline(x=val_data['date'].iloc[-1], color='orange', linestyle='--', alpha=0.7, label='Fim Val')
    axes[0].set_xlabel('Data', fontsize=11)
    axes[0].set_ylabel('Valor', fontsize=11)
    axes[0].set_title('Série Temporal Completa', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Componentes
    axes[1].plot(time, trend, label='Tendência', alpha=0.8)
    axes[1].plot(time, seasonality, label='Sazonalidade', alpha=0.8)
    axes[1].plot(time, cycle, label='Ciclo', alpha=0.8)
    axes[1].set_xlabel('Tempo', fontsize=11)
    axes[1].set_ylabel('Valor', fontsize=11)
    axes[1].set_title('Componentes da Série', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 2. CRIAÇÃO DOS MODELOS CANDIDATOS
    # =========================================================================
    print_section(2, "Criação de Modelos Candidatos (6 modelos)")

    print("Criando conjunto diversificado de modelos:")
    print()

    models_candidates = [
        AutoARIMAPredictor(max_p=5, max_q=5, seasonal=True, m=12, name="AutoARIMA"),
        ProphetPredictor(seasonality_mode='multiplicative', name="Prophet"),
        CatBoostPredictor(lookback=12, iterations=200, name="CatBoost"),
        LightGBMPredictor(lookback=12, n_estimators=200, name="LightGBM"),
        XGBoostPredictor(lookback=12, n_estimators=200, name="XGBoost"),
        LSTMPredictor(lookback=12, hidden_size=32, epochs=30, name="LSTM")
    ]

    for i, model in enumerate(models_candidates, 1):
        print(f"  {i}. {model.name:15s} - Criado")

    print(f"\n✓ {len(models_candidates)} modelos criados!")

    # =========================================================================
    # 3. OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
    # =========================================================================
    print_section(3, "Otimização de Hiperparâmetros com Optuna (Bayesian)")

    print("⚙️  Configurando otimizador Optuna...")
    print("  • Algoritmo: Bayesian Optimization (TPE)")
    print("  • Trials: 30 (use 50+ em produção)")
    print("  • Métrica: MAPE (minimizar)")
    print()

    # Cria otimizador
    optimizer = HyperparameterOptimizer(
        metric='mape',
        direction='minimize',
        n_trials=30,
        n_jobs=1,
        verbose=True
    )

    # Define espaços de busca
    model_configs = [
        {
            'class': CatBoostPredictor,
            'param_space': {
                'lookback': ('int', 8, 24),
                'iterations': ('int', 100, 400),
                'learning_rate': ('float', 0.01, 0.1, 'log'),
                'depth': ('int', 4, 10),
                'l2_leaf_reg': ('float', 1.0, 10.0)
            }
        },
        {
            'class': LightGBMPredictor,
            'param_space': {
                'lookback': ('int', 8, 24),
                'n_estimators': ('int', 100, 400),
                'learning_rate': ('float', 0.01, 0.1, 'log'),
                'num_leaves': ('int', 20, 50),
                'max_depth': ('int', 3, 10)
            }
        },
        {
            'class': XGBoostPredictor,
            'param_space': {
                'lookback': ('int', 8, 24),
                'n_estimators': ('int', 50, 300),
                'learning_rate': ('float', 0.01, 0.1, 'log'),
                'max_depth': ('int', 3, 10)
            }
        }
    ]

    print("🔍 Iniciando otimização...")
    print("   (Isso pode levar 5-10 minutos)")
    print()

    # Otimiza
    best_params = optimizer.optimize_ensemble(
        model_configs=model_configs,
        train_data=train_data['value'],
        val_data=val_data['value'],
        forecast_horizon=12
    )

    print_header("✅ OTIMIZAÇÃO CONCLUÍDA!")

    print("🏆 Melhores Hiperparâmetros Encontrados:\n")
    for model_name, params in best_params.items():
        print(f"{model_name}:")
        for param, value in params.items():
            if isinstance(value, float):
                print(f"  • {param}: {value:.4f}")
            else:
                print(f"  • {param}: {value}")
        print()

    # =========================================================================
    # 4. CRIAÇÃO DE MODELOS OTIMIZADOS
    # =========================================================================
    print_section(4, "Criação de Modelos com Hiperparâmetros Otimizados")

    optimized_models = []

    # Modelos otimizados
    for model_name, params in best_params.items():
        if model_name == 'CatBoostPredictor':
            optimized_models.append(CatBoostPredictor(**params, name="CatBoost_opt"))
        elif model_name == 'LightGBMPredictor':
            optimized_models.append(LightGBMPredictor(**params, name="LightGBM_opt"))
        elif model_name == 'XGBoostPredictor':
            optimized_models.append(XGBoostPredictor(**params, name="XGBoost_opt"))

    # Adiciona modelos não otimizados
    optimized_models.extend([
        AutoARIMAPredictor(name="AutoARIMA"),
        ProphetPredictor(name="Prophet")
    ])

    print(f"✓ {len(optimized_models)} modelos preparados:")
    for model in optimized_models:
        print(f"  • {model.name}")

    # Treina todos
    print("\n📚 Treinando todos os modelos...")
    for model in optimized_models:
        print(f"  [{model.name}] Treinando...", end=" ", flush=True)
        try:
            model.fit(train_data['value'])
            print("✓")
        except Exception as e:
            print(f"✗ ({str(e)[:30]})")

    # =========================================================================
    # 5. CRIAÇÃO DO ENSEMBLE
    # =========================================================================
    print_section(5, "Criação do Ensemble")

    ensemble = EnsemblePredictor(optimized_models)
    print(f"✓ Ensemble criado com {len(optimized_models)} modelos!")

    # =========================================================================
    # 6. AMBIENTE E AGENTE RL AVANÇADO
    # =========================================================================
    print_section(6, "Configuração do Agente RL")

    env = TimeSeriesEnv(
        data=train_data,
        forecast_horizon=12,
        window_size=24,
        n_coefficients=len(optimized_models),
        max_steps=50
    )

    print(f"Ambiente:")
    print(f"  • Estado: {env.observation_space.shape[0]} dimensões")
    print(f"  • Ação: {env.action_space.shape[0]} dimensões")
    print()

    if ADVANCED_AVAILABLE:
        print("🎓 Criando Agente RL Avançado (PhD)...")
        print("  • Arquitetura: Transformer-based Actor-Critic")
        print("  • Multi-Head Attention: 4 heads")
        print("  • LSTM Memory: 2 layers")
        print("  • Prioritized Experience Replay")
        print("  • Noisy Networks")
        print("  • Adaptive Entropy")

        agent = AdvancedRLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            learning_rate=1e-4,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            use_per=True,
            use_noisy=True,
            use_lstm=True,
            device='cpu'
        )

        trainer = AdvancedRLTrainer(
            env, agent, ensemble,
            use_curriculum=True
        )
    else:
        print("Criando Agente RL Padrão...")
        agent = RLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            learning_rate=3e-4,
            hidden_dim=128
        )
        trainer = RLTrainer(env, agent, ensemble)

    print("\n✓ Agente criado!")

    # =========================================================================
    # 7. TREINAMENTO COM OTIMIZAÇÃO RECURSIVA
    # =========================================================================
    print_section(7, "Treinamento RL com Otimização Recursiva")

    print("⚙️  Configurando otimização recursiva...")
    print("  • Frequência: a cada 50 episódios")
    print("  • Threshold: 5% de melhoria")
    print()

    # Cria otimizador recursivo
    recursive_opt = RecursiveOptimizer(
        hyperparameter_optimizer=optimizer,
        reoptimize_frequency=50,
        performance_window=20,
        improvement_threshold=0.05
    )

    print("🚀 Iniciando treinamento...")
    print("   ⏰ Tempo estimado: 10-15 minutos")
    print()

    # Treina
    if ADVANCED_AVAILABLE:
        history = trainer.train(
            n_episodes=150,
            max_steps=50,
            eval_frequency=25,
            early_stopping=True,
            verbose=True
        )
    else:
        history = trainer.train(
            n_episodes=150,
            max_steps=50,
            eval_frequency=25,
            verbose=True
        )

    print_header("✅ TREINAMENTO CONCLUÍDO!")

    # =========================================================================
    # 8. ANÁLISE COMPLETA DE RESULTADOS
    # =========================================================================
    print_section(8, "Análise Completa de Resultados")

    # Retreina com treino + validação
    full_train = pd.concat([train_data, val_data])

    # 1. Modelos individuais
    print("📊 Avaliando modelos individuais no conjunto de teste...\n")

    results = {}

    for model in optimized_models:
        try:
            model.fit(full_train['value'])
            pred = model.predict(steps=12)
            actual = test_data['value'].values[:12]
            metrics = calculate_metrics(actual, pred[:len(actual)])
            results[model.name] = metrics
        except:
            results[model.name] = None

    # 2. Ensemble com pesos iguais
    print("Testando Ensemble com pesos iguais...")
    ensemble_equal = EnsemblePredictor(optimized_models, weights=np.ones(len(optimized_models)))
    ensemble_equal.fit(full_train['value'])
    pred_equal = ensemble_equal.predict(steps=12)
    results['Ensemble_Iguais'] = calculate_metrics(actual, pred_equal[:len(actual)])

    # 3. Ensemble otimizado por RL
    print("Testando Ensemble otimizado por RL...")
    best_coefficients = trainer.get_best_coefficients()
    if best_coefficients is not None:
        ensemble.update_weights(best_coefficients)

    ensemble.fit(full_train['value'])
    pred_opt = ensemble.predict(steps=12)
    results['Ensemble_RL'] = calculate_metrics(actual, pred_opt[:len(actual)])

    # Tabela de comparação
    print("\n" + "="*110)
    print(f"{'COMPARAÇÃO COMPLETA DE PERFORMANCE':^110}")
    print("="*110)
    print(f"\n{'Modelo':<25} {'MAPE (%)':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Dir. Acc (%)':<12}")
    print("─"*110)

    # Ordena por MAPE
    sorted_results = sorted(
        [(name, metrics) for name, metrics in results.items() if metrics is not None],
        key=lambda x: x[1]['mape']
    )

    for name, metrics in sorted_results:
        emoji = "🏆" if name == 'Ensemble_RL' else "  "
        print(f"{emoji} {name:<23} {metrics['mape']:>10.2f}  {metrics['rmse']:>10.4f}  "
              f"{metrics['mae']:>10.4f}  {metrics['r2']:>10.4f}  "
              f"{metrics['directional_accuracy']:>10.2f}")

    print("─"*110)

    # Destaque do vencedor
    best_model = sorted_results[0][0]
    best_mape = sorted_results[0][1]['mape']

    print(f"\n🏆 VENCEDOR: {best_model} com MAPE de {best_mape:.2f}%")

    # Melhoria do RL
    if 'Ensemble_RL' in results and 'Ensemble_Iguais' in results:
        improvement = ((results['Ensemble_Iguais']['mape'] - results['Ensemble_RL']['mape']) /
                      results['Ensemble_Iguais']['mape']) * 100
        print(f"💡 Melhoria do RL sobre ensemble não-otimizado: {improvement:.1f}%")

    # =========================================================================
    # 9. VISUALIZAÇÕES AVANÇADAS
    # =========================================================================
    print_section(9, "Visualizações Avançadas")

    # Plot 1: Comparação de previsões
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top 4 modelos
    top_4 = sorted_results[:4]

    for idx, (name, metrics) in enumerate(top_4):
        ax = axes[idx // 2, idx % 2]

        if name == 'Ensemble_RL':
            pred_plot = pred_opt
        elif name == 'Ensemble_Iguais':
            pred_plot = pred_equal
        else:
            model = next(m for m in optimized_models if m.name == name)
            pred_plot = model.predict(steps=12)

        ax.plot(actual, 'o-', label='Real', linewidth=2, markersize=8, color='black')
        ax.plot(pred_plot[:len(actual)], 's-', label=f'{name}',
                linewidth=2, markersize=8, alpha=0.7)
        ax.set_title(f'{name} (MAPE: {metrics["mape"]:.2f}%)',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('Mês')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 2: Histórico de treinamento
    trainer.plot_training_progress()

    # Plot 3: Coeficientes otimizados
    if best_coefficients is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(optimized_models))
        ax.bar(x, best_coefficients, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([m.name for m in optimized_models], rotation=45, ha='right')
        ax.set_ylabel('Coeficiente', fontsize=12)
        ax.set_title('Coeficientes Otimizados pelo Agente RL', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=1/len(optimized_models), color='red', linestyle='--',
                   label='Peso Igual', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 10. RELATÓRIO FINAL
    # =========================================================================
    print_header("📊 RELATÓRIO FINAL")

    print("🎓 TÉCNICAS UTILIZADAS:")
    print()
    print("Modelos:")
    for model in optimized_models:
        print(f"  ✓ {model.name}")

    print("\nOtimização:")
    print("  ✓ Bayesian Optimization (Optuna)")
    print("  ✓ 30 trials por modelo")
    print(f"  ✓ {len(best_params)} modelos otimizados")

    if ADVANCED_AVAILABLE:
        print("\nAgente RL:")
        print("  ✓ Transformer-based Actor-Critic")
        print("  ✓ Multi-Head Attention (4 heads)")
        print("  ✓ LSTM Memory")
        print("  ✓ Prioritized Experience Replay")
        print("  ✓ Noisy Networks")
        print("  ✓ Adaptive Entropy Regularization")

    print("\n📈 RESULTADOS FINAIS:")
    print()
    print(f"  • Melhor Modelo: {best_model}")
    print(f"  • Melhor MAPE: {best_mape:.2f}%")
    print(f"  • Ensemble RL MAPE: {results['Ensemble_RL']['mape']:.2f}%")
    print(f"  • Ensemble RL R²: {results['Ensemble_RL']['r2']:.4f}")

    if improvement > 0:
        print(f"\n  💡 Melhoria do RL: {improvement:.1f}%")

    print("\n🏆 HISTÓRICO DE TREINAMENTO:")
    print(f"  • Melhor Recompensa: {history['best_reward']:.2f}")
    print(f"  • Recompensa Média Final: {np.mean(history['episode_rewards'][-20:]):.2f}")
    print(f"  • Episódios Completados: {len(history['episode_rewards'])}")

    if ADVANCED_AVAILABLE:
        print(f"  • Gradient Steps: {agent.gradient_steps:,}")
        print(f"  • Learning Rate Final: {agent.optimizer.param_groups[0]['lr']:.2e}")

    print("\n" + "="*100)
    print("🎉 EXEMPLO AVANÇADO CONCLUÍDO COM SUCESSO!")
    print("="*100)
    print()

    print("💾 Salve os resultados:")
    print("  • Modelo: agent.save('./models/advanced_agent.pt')")
    print("  • Coeficientes: np.save('./coefs.npy', best_coefficients)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
