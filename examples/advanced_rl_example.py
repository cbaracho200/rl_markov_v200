"""
Exemplo de uso do Agente RL Avançado (Nível PhD).

Demonstra todas as técnicas avançadas implementadas:
- Transformer-based Actor-Critic
- Multi-Head Attention
- Prioritized Experience Replay
- Noisy Networks
- Dueling Architecture
- LSTM Memory
- Ensemble Critics
- Adaptive Entropy Regularization
- Learning Rate Scheduling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importações do framework
from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent_advanced import AdvancedRLAgent
from src.training.trainer_advanced import AdvancedRLTrainer
from src.models.ensemble_predictor import EnsemblePredictor


def generate_sample_data(n_points: int = 200) -> pd.DataFrame:
    """
    Gera dados sintéticos de série temporal com múltiplos padrões.

    Args:
        n_points: Número de pontos

    Returns:
        DataFrame com série temporal
    """
    dates = pd.date_range(start='2010-01-01', periods=n_points, freq='M')

    # Componentes da série
    trend = np.linspace(100, 150, n_points)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_points) / 12)
    cycle = 15 * np.sin(2 * np.pi * np.arange(n_points) / 48)  # Ciclo de 4 anos
    noise = np.random.normal(0, 3, n_points)

    # Série temporal completa
    values = trend + seasonality + cycle + noise

    # Adiciona evento extremo (crise)
    if n_points > 100:
        values[100:110] *= 0.9  # Queda de 10%

    return pd.DataFrame({
        'date': dates,
        'value': values
    })


def main():
    """
    Função principal demonstrando o uso do agente avançado.
    """
    print("="*80)
    print("🎓 DEMONSTRAÇÃO: Agente RL Avançado (Nível PhD)")
    print("="*80)
    print()

    # 1. Preparação dos dados
    print("📊 1. Preparando dados...")
    data = generate_sample_data(n_points=200)
    print(f"   ✓ Dados gerados: {len(data)} pontos")
    print(f"   ✓ Período: {data['date'].min()} a {data['date'].max()}")
    print()

    # 2. Configuração do ambiente
    print("🏗️  2. Configurando ambiente de RL...")
    env = TimeSeriesEnv(
        data=data,
        forecast_horizon=12,  # 12 meses à frente
        window_size=24,       # Janela de 24 meses
        n_coefficients=10,    # 10 coeficientes para otimizar
        max_steps=50
    )
    print(f"   ✓ Ambiente criado")
    print(f"   ✓ Espaço de observação: {env.observation_space.shape}")
    print(f"   ✓ Espaço de ação: {env.action_space.shape}")
    print()

    # 3. Criação do agente avançado
    print("🤖 3. Criando Agente RL Avançado...")
    agent = AdvancedRLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_dim=512,        # Rede grande
        num_heads=8,           # 8 cabeças de atenção
        num_layers=3,          # 3 camadas de transformer
        use_per=True,          # Prioritized Experience Replay
        use_noisy=True,        # Noisy Networks
        use_lstm=True,         # LSTM para memória
        buffer_size=100000,    # Buffer grande
        device='cpu'           # Use 'cuda' se tiver GPU
    )
    print(f"   ✓ Agente criado com {sum(p.numel() for p in agent.policy.parameters()):,} parâmetros")
    print(f"   ✓ Arquitetura: Transformer com {agent.policy.num_heads} heads")
    print(f"   ✓ Hidden dim: {agent.policy.hidden_dim}")
    print()

    # 4. Configuração do treinamento
    print("🎯 4. Configurando treinamento...")
    trainer = AdvancedRLTrainer(
        env=env,
        agent=agent,
        log_dir='./logs_advanced',
        checkpoint_dir='./checkpoints_advanced',
        use_curriculum=True  # Curriculum learning
    )
    print(f"   ✓ Trainer configurado")
    print(f"   ✓ Curriculum Learning: Ativado")
    print(f"   ✓ Early Stopping: Ativado")
    print()

    # 5. Treinamento
    print("🚀 5. Iniciando treinamento...")
    print()

    history = trainer.train(
        n_episodes=200,        # Número de episódios (use mais em produção)
        max_steps=50,          # Passos por episódio
        eval_frequency=25,     # Avalia a cada 25 episódios
        save_frequency=50,     # Salva a cada 50 episódios
        early_stopping=True,   # Para se não melhorar
        verbose=True
    )

    print()
    print("="*80)
    print("✅ Treinamento concluído!")
    print("="*80)
    print()

    # 6. Avaliação final
    print("📈 6. Avaliando agente treinado...")
    results = trainer.evaluate(n_episodes=10, deterministic=True, verbose=True)

    # 7. Análise de resultados
    print("📊 7. Análise de Resultados:")
    print(f"   • Melhor recompensa: {history['best_reward']:.2f}")
    print(f"   • Recompensa média final: {np.mean(agent.episode_rewards):.2f}")
    print(f"   • Total de gradient steps: {agent.gradient_steps:,}")
    print(f"   • Learning rate final: {agent.optimizer.param_groups[0]['lr']:.2e}")
    print(f"   • Entropy coef final: {np.exp(agent.log_entropy_coef.detach().cpu().numpy()):.3f}")
    print()

    if 'mape' in results:
        print("📉 Métricas de Previsão:")
        print(f"   • MAPE: {results['mape']:.2f}%")
        print(f"   • RMSE: {results['rmse']:.4f}")
        print(f"   • MAE: {results['mae']:.4f}")
        print()

    # 8. Visualização
    print("📊 8. Gerando visualizações...")
    trainer.plot_training_progress(save_path='./training_advanced_progress.png')
    print()

    # 9. Salva modelo final
    print("💾 9. Salvando modelo final...")
    agent.save('./models/advanced_rl_final.pt')
    print("   ✓ Modelo salvo em ./models/advanced_rl_final.pt")
    print()

    print("="*80)
    print("🎉 Demonstração completa!")
    print("="*80)
    print()
    print("📚 Técnicas Implementadas:")
    print("   ✓ Transformer-based Actor-Critic com Multi-Head Attention")
    print("   ✓ Prioritized Experience Replay (PER)")
    print("   ✓ Noisy Networks para exploração adaptativa")
    print("   ✓ Dueling Architecture")
    print("   ✓ LSTM para memória temporal")
    print("   ✓ Ensemble de 3 Critics")
    print("   ✓ Adaptive Entropy Regularization")
    print("   ✓ Learning Rate Scheduling com Warmup")
    print("   ✓ Curriculum Learning")
    print("   ✓ Early Stopping")
    print("   ✓ Gradient Clipping")
    print("   ✓ Weight Decay (L2 Regularization)")
    print()
    print("🎓 Este é um modelo de nível PhD com técnicas state-of-the-art!")
    print()


if __name__ == "__main__":
    main()
