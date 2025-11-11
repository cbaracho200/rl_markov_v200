"""
Pipeline de treinamento para agente RL de previsão de séries temporais.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime

from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.models.ensemble_predictor import EnsemblePredictor


class RLTrainer:
    """
    Classe para treinamento do agente RL.

    Gerencia o loop de treinamento, logging de métricas,
    salvamento de checkpoints e avaliação.
    """

    def __init__(
        self,
        env: TimeSeriesEnv,
        agent: RLAgent,
        ensemble: Optional[EnsemblePredictor] = None,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Inicializa o trainer.

        Args:
            env: Ambiente de RL
            agent: Agente de RL
            ensemble: Ensemble de modelos (opcional)
            log_dir: Diretório para logs
            checkpoint_dir: Diretório para checkpoints
        """
        self.env = env
        self.agent = agent
        self.ensemble = ensemble

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Cria diretórios
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Métricas
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'best_reward': -np.inf,
            'best_coefficients': None
        }

    def train(
        self,
        n_episodes: int = 1000,
        max_steps: int = 100,
        update_frequency: int = 10,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Treina o agente RL.

        Args:
            n_episodes: Número de episódios de treinamento
            max_steps: Máximo de passos por episódio
            update_frequency: Frequência de atualização do agente (em passos)
            eval_frequency: Frequência de avaliação (em episódios)
            save_frequency: Frequência de salvamento (em episódios)
            verbose: Se True, exibe progresso

        Returns:
            Histórico de treinamento
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Iniciando treinamento do agente RL")
            print(f"{'='*80}")
            print(f"Episódios: {n_episodes}")
            print(f"Max steps por episódio: {max_steps}")
            print(f"Horizonte de previsão: {self.env.forecast_horizon} meses")
            print(f"{'='*80}\n")

        pbar = tqdm(range(n_episodes), desc="Treinamento") if verbose else range(n_episodes)

        for episode in pbar:
            # Reset ambiente
            state, info = self.env.reset()

            episode_reward = 0
            episode_length = 0

            # Episódio
            for step in range(max_steps):
                # Seleciona ação
                action = self.agent.select_action(state, deterministic=False)

                # Executa ação
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Armazena transição
                self.agent.store_transition(reward, terminated or truncated)

                # Atualiza métricas
                episode_reward += reward
                episode_length += 1

                # Atualiza estado
                state = next_state

                # Verifica se terminou
                if terminated or truncated:
                    break

            # Atualiza agente após cada episódio
            metrics = self.agent.update(state, n_epochs=10, batch_size=64)

            # Armazena recompensa do episódio
            self.agent.episode_rewards.append(episode_reward)

            # Salva métricas
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['mean_rewards'].append(np.mean(self.agent.episode_rewards))

            if metrics:
                self.training_history['policy_losses'].append(metrics.get('policy_loss', 0))
                self.training_history['value_losses'].append(metrics.get('value_loss', 0))
                self.training_history['entropies'].append(metrics.get('entropy', 0))

            # Atualiza melhor modelo
            if episode_reward > self.training_history['best_reward']:
                self.training_history['best_reward'] = episode_reward
                if 'coefficients' in info:
                    self.training_history['best_coefficients'] = info['coefficients']

            # Atualiza barra de progresso
            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'reward': f"{episode_reward:.2f}",
                    'mean_reward': f"{np.mean(self.agent.episode_rewards):.2f}",
                    'best': f"{self.training_history['best_reward']:.2f}"
                })

            # Avaliação periódica
            if (episode + 1) % eval_frequency == 0:
                eval_results = self.evaluate(n_episodes=5, deterministic=True, verbose=False)
                if verbose:
                    print(f"\n[Episódio {episode + 1}] Avaliação:")
                    print(f"  Recompensa média: {eval_results['mean_reward']:.2f}")
                    print(f"  Recompensa total: {eval_results['total_reward']:.2f}")

            # Salvamento periódico
            if (episode + 1) % save_frequency == 0:
                self.save_checkpoint(episode + 1)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Treinamento concluído!")
            print(f"Melhor recompensa: {self.training_history['best_reward']:.2f}")
            print(f"Recompensa média final: {np.mean(self.agent.episode_rewards):.2f}")
            print(f"{'='*80}\n")

        return self.training_history

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Avalia o agente treinado.

        Args:
            n_episodes: Número de episódios de avaliação
            deterministic: Se True, usa política determinística
            verbose: Se True, exibe resultados

        Returns:
            Resultados da avaliação
        """
        episode_rewards = []
        episode_lengths = []
        all_predictions = []
        all_actuals = []
        all_coefficients = []

        for episode in range(n_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            predictions = []
            actuals = []

            while True:
                # Ação determinística
                action = self.agent.select_action(state, deterministic=deterministic)

                # Executa
                next_state, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if 'prediction' in info:
                    predictions.append(info['prediction'])
                    actuals.append(info['actual'])

                state = next_state

                if terminated or truncated:
                    if 'coefficients' in info:
                        all_coefficients.append(info['coefficients'])
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)

        # Calcula métricas
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'total_reward': np.sum(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'predictions': all_predictions,
            'actuals': all_actuals,
            'coefficients': all_coefficients
        }

        # Métricas de previsão
        if len(all_predictions) > 0 and len(all_actuals) > 0:
            predictions_array = np.array(all_predictions)
            actuals_array = np.array(all_actuals)

            results['mse'] = np.mean((actuals_array - predictions_array) ** 2)
            results['rmse'] = np.sqrt(results['mse'])
            results['mae'] = np.mean(np.abs(actuals_array - predictions_array))
            results['mape'] = np.mean(np.abs((actuals_array - predictions_array) / (actuals_array + 1e-8))) * 100

        if verbose:
            print(f"\n{'='*60}")
            print(f"Resultados da Avaliação ({n_episodes} episódios)")
            print(f"{'='*60}")
            print(f"Recompensa média: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Recompensa total: {results['total_reward']:.2f}")
            print(f"Comprimento médio: {results['mean_length']:.1f}")
            if 'mape' in results:
                print(f"\nMétricas de Previsão:")
                print(f"  MAPE: {results['mape']:.2f}%")
                print(f"  RMSE: {results['rmse']:.4f}")
                print(f"  MAE: {results['mae']:.4f}")
            print(f"{'='*60}\n")

        return results

    def save_checkpoint(self, episode: int):
        """
        Salva checkpoint do treinamento.

        Args:
            episode: Número do episódio
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_ep{episode}_{timestamp}.pt"
        )

        # Salva agente
        self.agent.save(checkpoint_path)

        # Salva histórico
        history_path = os.path.join(
            self.checkpoint_dir,
            f"history_ep{episode}_{timestamp}.json"
        )

        with open(history_path, 'w') as f:
            # Converte arrays numpy para listas
            history_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.training_history.items()
                if k != 'best_coefficients'
            }
            json.dump(history_serializable, f, indent=2)

        print(f"\n✓ Checkpoint salvo: {checkpoint_path}")

    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Plota progresso do treinamento.

        Args:
            save_path: Caminho para salvar figura (opcional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Progresso do Treinamento RL', fontsize=16, fontweight='bold')

        # Recompensas por episódio
        axes[0, 0].plot(self.training_history['episode_rewards'], alpha=0.6, label='Por episódio')
        if len(self.training_history['mean_rewards']) > 0:
            axes[0, 0].plot(self.training_history['mean_rewards'], linewidth=2, label='Média móvel')
        axes[0, 0].set_xlabel('Episódio')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].set_title('Recompensas')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Perdas
        if len(self.training_history['policy_losses']) > 0:
            axes[0, 1].plot(self.training_history['policy_losses'], label='Policy Loss')
            axes[0, 1].plot(self.training_history['value_losses'], label='Value Loss')
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Perdas de Treinamento')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Comprimento dos episódios
        axes[1, 0].plot(self.training_history['episode_lengths'])
        axes[1, 0].set_xlabel('Episódio')
        axes[1, 0].set_ylabel('Passos')
        axes[1, 0].set_title('Comprimento dos Episódios')
        axes[1, 0].grid(True, alpha=0.3)

        # Entropia
        if len(self.training_history['entropies']) > 0:
            axes[1, 1].plot(self.training_history['entropies'])
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Entropia')
            axes[1, 1].set_title('Entropia da Política')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Gráfico salvo: {save_path}")

        plt.show()

    def get_best_coefficients(self) -> np.ndarray:
        """
        Retorna os melhores coeficientes encontrados.

        Returns:
            Array com os melhores coeficientes
        """
        return self.training_history.get('best_coefficients', None)
