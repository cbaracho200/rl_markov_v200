"""
Pipeline de treinamento avançado para agente RL de nível PhD.

Suporta o AdvancedRLAgent com Transformer, PER, e outras técnicas avançadas.
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
from src.agents.rl_agent_advanced import AdvancedRLAgent
from src.models.ensemble_predictor import EnsemblePredictor


class AdvancedRLTrainer:
    """
    Trainer avançado para AdvancedRLAgent com técnicas de nível PhD.

    Inclui:
    - Curriculum Learning
    - Early Stopping
    - Adaptive Training
    - Advanced Metrics Tracking
    """

    def __init__(
        self,
        env: TimeSeriesEnv,
        agent: AdvancedRLAgent,
        ensemble: Optional[EnsemblePredictor] = None,
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        use_curriculum: bool = True
    ):
        """
        Inicializa o trainer avançado.

        Args:
            env: Ambiente de RL
            agent: Agente RL avançado
            ensemble: Ensemble de modelos (opcional)
            log_dir: Diretório para logs
            checkpoint_dir: Diretório para checkpoints
            use_curriculum: Se True, usa curriculum learning
        """
        self.env = env
        self.agent = agent
        self.ensemble = ensemble
        self.use_curriculum = use_curriculum

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Cria diretórios
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Métricas avançadas
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'entropy_coefs': [],
            'learning_rates': [],
            'gradient_norms': [],
            'best_reward': -np.inf,
            'best_coefficients': None,
            'curriculum_stage': 0
        }

        # Early stopping
        self.patience = 50
        self.best_eval_reward = -np.inf
        self.patience_counter = 0

    def train(
        self,
        n_episodes: int = 1000,
        max_steps: int = 100,
        eval_frequency: int = 25,
        save_frequency: int = 50,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Treina o agente RL avançado.

        Args:
            n_episodes: Número de episódios de treinamento
            max_steps: Máximo de passos por episódio
            eval_frequency: Frequência de avaliação (em episódios)
            save_frequency: Frequência de salvamento (em episódios)
            early_stopping: Se True, usa early stopping
            verbose: Se True, exibe progresso

        Returns:
            Histórico de treinamento
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"🎓 Treinamento Avançado do Agente RL (Nível PhD)")
            print(f"{'='*80}")
            print(f"Episódios: {n_episodes}")
            print(f"Max steps por episódio: {max_steps}")
            print(f"Horizonte de previsão: {self.env.forecast_horizon} meses")
            print(f"Técnicas ativadas:")
            print(f"  ✓ Transformer-based Actor-Critic")
            print(f"  ✓ Multi-Head Attention")
            print(f"  ✓ Prioritized Experience Replay")
            print(f"  ✓ Noisy Networks")
            print(f"  ✓ Dueling Architecture")
            print(f"  ✓ LSTM Memory")
            print(f"  ✓ Ensemble Critics")
            print(f"  ✓ Adaptive Entropy Regularization")
            print(f"  ✓ Learning Rate Scheduling")
            if self.use_curriculum:
                print(f"  ✓ Curriculum Learning")
            if early_stopping:
                print(f"  ✓ Early Stopping (patience={self.patience})")
            print(f"{'='*80}\n")

        pbar = tqdm(range(n_episodes), desc="🚀 Treinamento Avançado") if verbose else range(n_episodes)

        for episode in pbar:
            # Curriculum Learning: ajusta dificuldade
            if self.use_curriculum:
                self._adjust_curriculum(episode, n_episodes)

            # Reset ambiente e LSTM hidden state
            state, info = self.env.reset()
            self.agent.policy.reset_lstm_hidden()

            episode_reward = 0
            episode_length = 0

            # Episódio
            for step in range(max_steps):
                # Seleciona ação (com exploração via Noisy Networks)
                action, action_info = self.agent.select_action(state, deterministic=False)

                # Executa ação
                next_state, reward, terminated, truncated, step_info = self.env.step(action)

                # Armazena transição no buffer (PER ou normal)
                self.agent.store_transition(
                    state, action, reward, next_state,
                    terminated or truncated, action_info
                )

                # Atualiza métricas
                episode_reward += reward
                episode_length += 1

                # Atualiza estado
                state = next_state

                # Verifica se terminou
                if terminated or truncated:
                    break

            # Atualiza agente (usa PER para sampling)
            metrics = self.agent.update(n_epochs=10, batch_size=64)

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
                self.training_history['entropy_coefs'].append(metrics.get('entropy_coef', 0.01))
                self.training_history['learning_rates'].append(metrics.get('learning_rate', 0))

            # Atualiza melhor modelo
            if episode_reward > self.training_history['best_reward']:
                self.training_history['best_reward'] = episode_reward
                if 'coefficients' in step_info:
                    self.training_history['best_coefficients'] = step_info['coefficients']

            # Atualiza barra de progresso
            if verbose and hasattr(pbar, 'set_postfix'):
                postfix = {
                    'reward': f"{episode_reward:.2f}",
                    'mean': f"{np.mean(self.agent.episode_rewards):.2f}",
                    'best': f"{self.training_history['best_reward']:.2f}",
                }
                if metrics:
                    postfix['lr'] = f"{metrics.get('learning_rate', 0):.2e}"
                    postfix['ent_coef'] = f"{metrics.get('entropy_coef', 0):.3f}"
                pbar.set_postfix(postfix)

            # Avaliação periódica
            if (episode + 1) % eval_frequency == 0:
                eval_results = self.evaluate(n_episodes=5, deterministic=True, verbose=False)

                if verbose:
                    print(f"\n📊 [Episódio {episode + 1}] Avaliação:")
                    print(f"  Recompensa média: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                    print(f"  Buffer size: {len(self.agent.replay_buffer)}")
                    if 'mape' in eval_results:
                        print(f"  MAPE: {eval_results['mape']:.2f}%")

                # Early stopping
                if early_stopping:
                    if eval_results['mean_reward'] > self.best_eval_reward:
                        self.best_eval_reward = eval_results['mean_reward']
                        self.patience_counter = 0
                        # Salva melhor modelo
                        self.save_checkpoint(episode + 1, is_best=True)
                    else:
                        self.patience_counter += 1

                        if self.patience_counter >= self.patience:
                            if verbose:
                                print(f"\n⚠️  Early stopping acionado após {self.patience} avaliações sem melhora")
                            break

            # Salvamento periódico
            if (episode + 1) % save_frequency == 0:
                self.save_checkpoint(episode + 1)

        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ Treinamento concluído!")
            print(f"Melhor recompensa: {self.training_history['best_reward']:.2f}")
            print(f"Recompensa média final: {np.mean(self.agent.episode_rewards):.2f}")
            print(f"Total de gradient steps: {self.agent.gradient_steps}")
            print(f"{'='*80}\n")

        return self.training_history

    def _adjust_curriculum(self, episode: int, total_episodes: int):
        """
        Ajusta dificuldade do curriculum learning.

        Args:
            episode: Episódio atual
            total_episodes: Total de episódios
        """
        # Aumenta dificuldade gradualmente
        progress = episode / total_episodes

        # Estágios: easy (0-0.3), medium (0.3-0.6), hard (0.6-1.0)
        if progress < 0.3:
            stage = 0  # Easy
        elif progress < 0.6:
            stage = 1  # Medium
        else:
            stage = 2  # Hard

        if stage != self.training_history['curriculum_stage']:
            self.training_history['curriculum_stage'] = stage
            # Aqui você pode ajustar parâmetros do ambiente baseado no stage

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
            self.agent.policy.reset_lstm_hidden()

            episode_reward = 0
            episode_length = 0
            predictions = []
            actuals = []

            while True:
                # Ação determinística
                action, _ = self.agent.select_action(state, deterministic=deterministic)

                # Executa
                next_state, reward, terminated, truncated, step_info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if 'prediction' in step_info:
                    predictions.append(step_info['prediction'])
                    actuals.append(step_info['actual'])

                state = next_state

                if terminated or truncated:
                    if 'coefficients' in step_info:
                        all_coefficients.append(step_info['coefficients'])
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
            print(f"📊 Resultados da Avaliação ({n_episodes} episódios)")
            print(f"{'='*60}")
            print(f"Recompensa média: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"Recompensa total: {results['total_reward']:.2f}")
            print(f"Comprimento médio: {results['mean_length']:.1f}")
            if 'mape' in results:
                print(f"\n📈 Métricas de Previsão:")
                print(f"  MAPE: {results['mape']:.2f}%")
                print(f"  RMSE: {results['rmse']:.4f}")
                print(f"  MAE: {results['mae']:.4f}")
            print(f"{'='*60}\n")

        return results

    def save_checkpoint(self, episode: int, is_best: bool = False):
        """
        Salva checkpoint do treinamento.

        Args:
            episode: Número do episódio
            is_best: Se True, marca como melhor modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "best_" if is_best else ""

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{prefix}checkpoint_ep{episode}_{timestamp}.pt"
        )

        # Salva agente
        self.agent.save(checkpoint_path)

        # Salva histórico
        history_path = os.path.join(
            self.checkpoint_dir,
            f"{prefix}history_ep{episode}_{timestamp}.json"
        )

        with open(history_path, 'w') as f:
            # Converte arrays numpy para listas
            history_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.training_history.items()
                if k != 'best_coefficients'
            }
            json.dump(history_serializable, f, indent=2)

        print(f"\n✓ {'Melhor modelo' if is_best else 'Checkpoint'} salvo: {checkpoint_path}")

    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Plota progresso do treinamento com métricas avançadas.

        Args:
            save_path: Caminho para salvar figura (opcional)
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('🎓 Progresso do Treinamento RL Avançado', fontsize=16, fontweight='bold')

        # Recompensas por episódio
        axes[0, 0].plot(self.training_history['episode_rewards'], alpha=0.6, label='Por episódio', linewidth=0.8)
        if len(self.training_history['mean_rewards']) > 0:
            axes[0, 0].plot(self.training_history['mean_rewards'], linewidth=2, label='Média móvel (100)', color='red')
        axes[0, 0].set_xlabel('Episódio')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].set_title('Recompensas ao Longo do Tempo')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Perdas
        if len(self.training_history['policy_losses']) > 0:
            axes[0, 1].plot(self.training_history['policy_losses'], label='Policy Loss', alpha=0.7)
            axes[0, 1].plot(self.training_history['value_losses'], label='Value Loss', alpha=0.7)
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Perdas de Treinamento')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')

        # Comprimento dos episódios
        axes[1, 0].plot(self.training_history['episode_lengths'], color='green', alpha=0.6)
        axes[1, 0].set_xlabel('Episódio')
        axes[1, 0].set_ylabel('Passos')
        axes[1, 0].set_title('Comprimento dos Episódios')
        axes[1, 0].grid(True, alpha=0.3)

        # Entropia
        if len(self.training_history['entropies']) > 0:
            axes[1, 1].plot(self.training_history['entropies'], color='purple', alpha=0.7)
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Entropia')
            axes[1, 1].set_title('Entropia da Política (Exploração)')
            axes[1, 1].grid(True, alpha=0.3)

        # Learning Rate
        if len(self.training_history['learning_rates']) > 0:
            axes[2, 0].plot(self.training_history['learning_rates'], color='orange', alpha=0.7)
            axes[2, 0].set_xlabel('Update')
            axes[2, 0].set_ylabel('Learning Rate')
            axes[2, 0].set_title('Taxa de Aprendizado (Scheduling)')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].set_yscale('log')

        # Coeficiente de Entropia Adaptativo
        if len(self.training_history['entropy_coefs']) > 0:
            axes[2, 1].plot(self.training_history['entropy_coefs'], color='brown', alpha=0.7)
            axes[2, 1].set_xlabel('Update')
            axes[2, 1].set_ylabel('Entropy Coefficient')
            axes[2, 1].set_title('Coeficiente de Entropia Adaptativo')
            axes[2, 1].grid(True, alpha=0.3)

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
