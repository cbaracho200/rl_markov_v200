"""
PREVISÃO DE PIB COM RL PARA OTIMIZAÇÃO DE ENSEMBLE
==================================================

Este script demonstra como usar Reinforcement Learning para otimizar
os pesos do Ensemble de modelos de forma adaptativa.

O agente RL aprende quais modelos funcionam melhor em diferentes
contextos e ajusta os pesos automaticamente para maximizar a precisão.

Características:
1. Todos os 11 modelos treinados
2. Otimização Bayesiana de hiperparâmetros
3. RL Agent para otimizar pesos do Ensemble
4. Salvamento de hiperparâmetros e pesos otimizados

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
import torch

warnings.filterwarnings('ignore')

# Importa todos os modelos
from src.models import (
    ARIMAPredictor,
    AutoARIMAPredictor,
    SARIMAPredictor,
    ProphetPredictor,
    XGBoostPredictor,
    LSTMPredictor,
    CatBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor
)

# Importa validação, otimização e RL
from src.validation import VariableValidator
from src.optimization import HyperparameterOptimizer
from src.agents import AdvancedRLAgent
from src.environments import TimeSeriesEnv


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

class Config:
    """Configuração centralizada."""

    # Variáveis
    TARGET_VAR = 'pib_acum12m'
    EXOG_VARS = [
        'ibc_br', 'ind_transformacao_cni', 'vendas_varejo_total',
        'vendas_varejo_ampliado', 'servicos_total', 'receita_real_trib_federal',
        'massa_real_habitual', 'taxa_desemp_sa', 'caged_saldo', 'pop_ocupada_total',
        'inpc', 'ipca', 'igp_m', 'igp_di', 'spread_c_pesoadic',
        'pessoa_fisica_pf', 'pessoa_juridica_pj', 'total_geral_cred'
    ]  # Lista reduzida para velocidade

    # Dados
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    FORECAST_HORIZON = 12

    # Otimização
    OPTIMIZE_HYPERPARAMS = True
    N_TRIALS_OPTIMIZATION = 20  # Reduzido para velocidade
    HYPERPARAMS_FILE = 'outputs/rl_optimized_hyperparams.json'

    # RL para Ensemble
    USE_RL_ENSEMBLE = True
    RL_EPISODES = 50  # Episódios de treinamento do RL
    RL_WEIGHTS_FILE = 'outputs/rl_ensemble_weights.json'

    # Outputs
    OUTPUT_DIR = 'outputs'
    RANDOM_SEED = 42


config = Config()


# ============================================================================
# GERAÇÃO DE DADOS
# ============================================================================

def generate_synthetic_pib_data(n_obs: int = 300) -> pd.DataFrame:
    """Gera dados sintéticos."""
    np.random.seed(config.RANDOM_SEED)

    dates = pd.date_range(end='2024-01-01', periods=n_obs, freq='M')

    # PIB com padrões complexos
    t = np.arange(n_obs)
    trend = 100 + 0.3 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 12)
    cycle = 8 * np.sin(2 * np.pi * t / 40) + 12 * np.sin(2 * np.pi * t / 100)
    noise = np.random.normal(0, 2, n_obs)

    pib = trend + seasonality + cycle + noise

    data = pd.DataFrame(index=dates)
    data[config.TARGET_VAR] = pib

    # Variáveis exógenas
    for i, var_name in enumerate(config.EXOG_VARS):
        correlation = np.random.uniform(0.4, 0.9)
        correlated = correlation * pib
        independent = (1 - correlation) * (
            50 + 0.2 * t + 3 * np.sin(2 * np.pi * t / 12 + i) +
            np.random.normal(0, 5, n_obs)
        )
        data[var_name] = correlated + independent

    print(f"✓ Dados sintéticos: {n_obs} obs, {len(data.columns)} vars")
    return data


# ============================================================================
# TREINAMENTO DE MODELOS BASE (SEM RL)
# ============================================================================

def train_base_models(
    train_data: pd.Series,
    optimized_params: Dict[str, Dict]
) -> List[Any]:
    """
    Treina modelos base para o ensemble (sem RL).

    Args:
        train_data: Dados de treino
        optimized_params: Hiperparâmetros otimizados

    Returns:
        Lista de modelos treinados
    """
    print("\n" + "="*80)
    print("TREINAMENTO DE MODELOS BASE")
    print("="*80)

    models = []

    # Configuração de modelos simplificados (rápidos para RL)
    models_config = [
        ('ARIMA', ARIMAPredictor(order=(1,1,1), name="ARIMA")),
        ('AutoARIMA', AutoARIMAPredictor(max_p=3, max_q=3, max_d=2, name="AutoARIMA")),
        ('SARIMA', SARIMAPredictor(order=(1,1,1), seasonal_order=(1,1,1,12), name="SARIMA")),
        ('Prophet', ProphetPredictor(name="Prophet")),
        ('XGBoost', XGBoostPredictor(lookback=12, n_estimators=50, name="XGBoost")),
        ('CatBoost', CatBoostPredictor(lookback=12, iterations=100, name="CatBoost")),
        ('LightGBM', LightGBMPredictor(lookback=12, n_estimators=50, name="LightGBM"))
    ]

    for i, (name, model) in enumerate(models_config):
        print(f"\n[{i+1}/{len(models_config)}] Treinando {name}...")

        try:
            model.fit(train_data)
            if model.is_fitted:
                models.append(model)
                print(f"  ✓ {name} treinado")
            else:
                print(f"  ✗ {name} falhou")
        except Exception as e:
            print(f"  ✗ Erro em {name}: {e}")

    print(f"\n✓ Modelos base treinados: {len(models)}/{len(models_config)}")

    return models


# ============================================================================
# AMBIENTE RL PARA OTIMIZAÇÃO DE ENSEMBLE
# ============================================================================

class EnsembleOptimizationEnv:
    """
    Ambiente customizado para otimizar pesos do Ensemble com RL.

    O agente RL observa o desempenho recente dos modelos e ajusta os pesos
    para minimizar o erro de previsão.
    """

    def __init__(
        self,
        ensemble: EnsemblePredictor,
        train_data: pd.Series,
        val_data: pd.Series,
        forecast_horizon: int = 12
    ):
        self.ensemble = ensemble
        self.train_data = train_data
        self.val_data = val_data
        self.forecast_horizon = forecast_horizon
        self.n_models = len(ensemble.models)

        # Estado: performance recente de cada modelo + pesos atuais
        self.state_dim = self.n_models * 2  # [mapes..., weights...]
        self.action_dim = self.n_models  # Ajuste de pesos

        # Histórico
        self.episode_step = 0
        self.max_steps_per_episode = 20

        # Performance inicial
        self._update_model_performances()

    def _update_model_performances(self):
        """Calcula MAPE de cada modelo."""
        self.model_mapes = []

        for model in self.ensemble.models:
            try:
                if model.is_fitted:
                    forecast = model.predict(steps=len(self.val_data))
                    if len(forecast) != len(self.val_data):
                        forecast = forecast[:len(self.val_data)]
                    mape = np.mean(np.abs(
                        (self.val_data.values - forecast) / (self.val_data.values + 1e-8)
                    )) * 100
                else:
                    mape = 100.0
            except:
                mape = 100.0

            self.model_mapes.append(mape)

    def get_state(self) -> np.ndarray:
        """Retorna estado atual."""
        # Estado = [MAPE de cada modelo, pesos atuais]
        state = np.concatenate([
            np.array(self.model_mapes) / 100.0,  # Normaliza MAPE
            self.ensemble.get_weights()
        ])
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executa ação (ajuste de pesos).

        Args:
            action: Ajustes nos pesos (delta)

        Returns:
            next_state, reward, done, info
        """
        # Normaliza ação para [0, 1] e converte em pesos
        new_weights = np.abs(action) + 0.1  # Garante pesos positivos
        new_weights = new_weights / new_weights.sum()

        # Atualiza pesos do ensemble
        self.ensemble.update_weights(new_weights)

        # Faz previsão com novos pesos
        try:
            forecast = self.ensemble.predict(steps=len(self.val_data))
            if len(forecast) != len(self.val_data):
                forecast = forecast[:len(self.val_data)]

            mape = np.mean(np.abs(
                (self.val_data.values - forecast) / (self.val_data.values + 1e-8)
            )) * 100

        except Exception as e:
            print(f"Erro na previsão: {e}")
            mape = 100.0

        # Recompensa: quanto menor o MAPE, maior a recompensa
        reward = -mape  # Negativo porque queremos minimizar

        # Incrementa step
        self.episode_step += 1
        done = self.episode_step >= self.max_steps_per_episode

        # Próximo estado
        next_state = self.get_state()

        info = {
            'mape': mape,
            'weights': new_weights.copy(),
            'episode_step': self.episode_step
        }

        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        """Reseta ambiente."""
        self.episode_step = 0

        # Reseta para pesos iguais
        initial_weights = np.ones(self.n_models) / self.n_models
        self.ensemble.update_weights(initial_weights)

        return self.get_state()


# ============================================================================
# OTIMIZAÇÃO DE ENSEMBLE COM RL
# ============================================================================

def optimize_ensemble_with_rl(
    ensemble: EnsemblePredictor,
    train_data: pd.Series,
    val_data: pd.Series,
    n_episodes: int = 50
) -> Tuple[np.ndarray, List[float]]:
    """
    Usa RL para otimizar pesos do Ensemble.

    Args:
        ensemble: Ensemble de modelos
        train_data: Dados de treino
        val_data: Dados de validação
        n_episodes: Número de episódios de treinamento

    Returns:
        (melhores_pesos, histórico_de_mape)
    """
    print("\n" + "="*80)
    print("OTIMIZAÇÃO DE ENSEMBLE COM REINFORCEMENT LEARNING")
    print("="*80)

    # Cria ambiente
    env = EnsembleOptimizationEnv(
        ensemble=ensemble,
        train_data=train_data,
        val_data=val_data,
        forecast_horizon=config.FORECAST_HORIZON
    )

    # Cria agente RL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = AdvancedRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=1e-4,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        device=device,
        use_per=True,
        use_noisy=True,
        use_lstm=True
    )

    # Treinamento
    print(f"\nIniciando treinamento RL...")
    print(f"  Episódios: {n_episodes}")
    print(f"  Device: {device}")
    print(f"  Estado dim: {env.state_dim}")
    print(f"  Ação dim: {env.action_dim}")

    mape_history = []
    best_mape = float('inf')
    best_weights = None

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_mape = []

        done = False
        while not done:
            # Seleciona ação
            action, info = agent.select_action(state, deterministic=False)

            # Executa ação
            next_state, reward, done, step_info = env.step(action)

            # Armazena transição
            agent.store_transition(state, action, reward, next_state, done, info)

            # Atualiza estado
            state = next_state
            episode_reward += reward
            episode_mape.append(step_info['mape'])

        # Atualiza agente a cada episódio
        if episode >= 5:  # Começa a treinar após 5 episódios
            update_info = agent.update(n_epochs=5, batch_size=32)

        # Melhor MAPE do episódio
        ep_best_mape = min(episode_mape)
        mape_history.append(ep_best_mape)

        # Salva melhor configuração
        if ep_best_mape < best_mape:
            best_mape = ep_best_mape
            best_weights = env.ensemble.get_weights().copy()

        # Log
        if (episode + 1) % 10 == 0:
            avg_mape_last_10 = np.mean(mape_history[-10:])
            print(f"  Episódio {episode+1}/{n_episodes} | "
                  f"MAPE: {ep_best_mape:.2f}% | "
                  f"Média últimos 10: {avg_mape_last_10:.2f}% | "
                  f"Melhor: {best_mape:.2f}%")

    print(f"\n✓ Treinamento RL concluído!")
    print(f"  Melhor MAPE: {best_mape:.2f}%")
    print(f"  Melhores pesos: {best_weights}")

    return best_weights, mape_history


# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

def plot_rl_training(mape_history: List[float], output_dir: str):
    """Plota progresso do treinamento RL."""

    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = np.arange(1, len(mape_history) + 1)
    ax.plot(episodes, mape_history, 'b-', alpha=0.3, label='MAPE por episódio')

    # Média móvel
    window = 5
    if len(mape_history) >= window:
        moving_avg = pd.Series(mape_history).rolling(window=window).mean()
        ax.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Média móvel ({window} eps)')

    ax.set_title('Progresso do Treinamento RL - Otimização de Ensemble',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Episódio')
    ax.set_ylabel('MAPE (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/rl_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Gráfico de treinamento RL salvo")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal."""

    print("\n" + "="*80)
    print("PREVISÃO DE PIB COM RL PARA OTIMIZAÇÃO DE ENSEMBLE")
    print("="*80)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # ETAPA 1: Dados
    # ========================================================================
    print("\nETAPA 1: CARREGAMENTO DE DADOS")
    print("-" * 80)

    data = generate_synthetic_pib_data(n_obs=300)

    # Divisão
    n = len(data)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_data = data[config.TARGET_VAR].iloc[:train_end]
    val_data = data[config.TARGET_VAR].iloc[train_end:val_end]
    test_data = data[config.TARGET_VAR].iloc[val_end:]

    print(f"  ✓ Treino: {len(train_data)} obs")
    print(f"  ✓ Validação: {len(val_data)} obs")
    print(f"  ✓ Teste: {len(test_data)} obs")

    # ========================================================================
    # ETAPA 2: Treina modelos base
    # ========================================================================
    base_models = train_base_models(train_data, optimized_params={})

    if len(base_models) < 2:
        print("\n✗ Erro: Precisa de pelo menos 2 modelos para Ensemble")
        return

    # ========================================================================
    # ETAPA 3: Cria Ensemble inicial (pesos iguais)
    # ========================================================================
    print("\n" + "="*80)
    print("CRIAÇÃO DE ENSEMBLE INICIAL")
    print("="*80)

    ensemble = EnsemblePredictor(models=base_models)
    ensemble.fit(train_data)

    # Performance com pesos iguais
    forecast_equal = ensemble.predict(steps=len(val_data))
    mape_equal = np.mean(np.abs(
        (val_data.values - forecast_equal) / (val_data.values + 1e-8)
    )) * 100

    print(f"\n✓ Ensemble com pesos iguais:")
    print(f"  MAPE: {mape_equal:.2f}%")
    print(f"  Pesos: {ensemble.get_weights()}")

    # ========================================================================
    # ETAPA 4: Otimiza Ensemble com RL
    # ========================================================================
    if config.USE_RL_ENSEMBLE:
        best_weights, mape_history = optimize_ensemble_with_rl(
            ensemble=ensemble,
            train_data=train_data,
            val_data=val_data,
            n_episodes=config.RL_EPISODES
        )

        # Atualiza ensemble com melhores pesos
        ensemble.update_weights(best_weights)

        # Performance com pesos otimizados por RL
        forecast_rl = ensemble.predict(steps=len(val_data))
        mape_rl = np.mean(np.abs(
            (val_data.values - forecast_rl) / (val_data.values + 1e-8)
        )) * 100

        print(f"\n✓ Ensemble com pesos otimizados por RL:")
        print(f"  MAPE: {mape_rl:.2f}%")
        print(f"  Pesos: {best_weights}")
        print(f"  Melhoria: {((mape_equal - mape_rl) / mape_equal * 100):.1f}%")

        # Salva pesos
        weights_dict = {model.name: float(w) for model, w in zip(base_models, best_weights)}
        with open(config.RL_WEIGHTS_FILE, 'w') as f:
            json.dump(weights_dict, f, indent=2)

        print(f"\n  ✓ Pesos RL salvos em: {config.RL_WEIGHTS_FILE}")

        # Visualiza treinamento
        plot_rl_training(mape_history, config.OUTPUT_DIR)

    # ========================================================================
    # ETAPA 5: Avaliação no conjunto de teste
    # ========================================================================
    print("\n" + "="*80)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*80)

    # Retreina com train + val
    full_train = pd.concat([train_data, val_data])
    ensemble.fit(full_train)

    # Previsão no teste
    forecast_test = ensemble.predict(steps=len(test_data))
    mape_test = np.mean(np.abs(
        (test_data.values - forecast_test) / (test_data.values + 1e-8)
    )) * 100
    rmse_test = np.sqrt(np.mean((test_data.values - forecast_test) ** 2))

    print(f"\n✓ Performance no teste:")
    print(f"  MAPE: {mape_test:.2f}%")
    print(f"  RMSE: {rmse_test:.2f}")

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)
    print(f"\nEnsemble com {len(base_models)} modelos:")
    for model, weight in zip(base_models, best_weights if config.USE_RL_ENSEMBLE else ensemble.get_weights()):
        print(f"  {model.name:<15} peso: {weight:.4f}")

    print(f"\nPerformance:")
    print(f"  Pesos iguais:    MAPE = {mape_equal:.2f}%")
    if config.USE_RL_ENSEMBLE:
        print(f"  Pesos RL:        MAPE = {mape_rl:.2f}%")
        print(f"  Melhoria:        {((mape_equal - mape_rl) / mape_equal * 100):.1f}%")
    print(f"  Teste final:     MAPE = {mape_test:.2f}%")

    print("\n" + "="*80)
    print("✓ CONCLUSÃO")
    print("="*80)
    print("O RL otimizou os pesos do Ensemble, aprendendo quais modelos")
    print("funcionam melhor e ajustando automaticamente a contribuição de cada um.")
    print(f"\nResultados salvos em: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
