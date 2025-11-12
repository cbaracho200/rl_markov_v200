"""
Agente de Reinforcement Learning para otimização de coeficientes de previsão.

Implementa PPO (Proximal Policy Optimization) para aprendizado de políticas
de ajuste de coeficientes em ensemble de modelos de séries temporais.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, List, Dict, Optional
import gymnasium as gym
from collections import deque
import json


class ActorCritic(nn.Module):
    """
    Rede neural Actor-Critic para PPO.

    Actor: Aprende a política (quais ações tomar)
    Critic: Estima o valor do estado (value function)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Inicializa a rede Actor-Critic.

        Args:
            state_dim: Dimensão do espaço de estados
            action_dim: Dimensão do espaço de ações
            hidden_dim: Tamanho das camadas ocultas
        """
        super(ActorCritic, self).__init__()

        # Camadas compartilhadas
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Actor (política)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Saída entre -1 e 1
        )

        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (função valor)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Inicialização de pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa pesos da rede."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass da rede.

        Args:
            state: Estado do ambiente

        Returns:
            distribution: Distribuição de probabilidade das ações
            value: Valor estimado do estado
        """
        shared_features = self.shared_layers(state)

        # Actor
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_logstd)
        distribution = Normal(action_mean, action_std)

        # Critic
        value = self.critic(shared_features)

        return distribution, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Seleciona uma ação baseada no estado.

        Args:
            state: Estado atual
            deterministic: Se True, retorna ação determinística (média)

        Returns:
            action: Ação selecionada
            log_prob: Log da probabilidade da ação
            value: Valor estimado do estado
        """
        distribution, value = self.forward(state)

        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Avalia ações tomadas em estados específicos.

        Args:
            states: Batch de estados
            actions: Batch de ações

        Returns:
            log_probs: Log probabilidades das ações
            values: Valores estimados dos estados
            entropy: Entropia da distribuição (para regularização)
        """
        distribution, values = self.forward(states)

        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy


class RLAgent:
    """
    Agente de Reinforcement Learning usando PPO.

    Este agente aprende a otimizar coeficientes de ensemble de modelos
    supervisionados para maximizar a precisão de previsões de séries temporais.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        device: str = 'cpu'
    ):
        """
        Inicializa o agente PPO.

        Args:
            state_dim: Dimensão do espaço de estados
            action_dim: Dimensão do espaço de ações
            learning_rate: Taxa de aprendizado
            gamma: Fator de desconto
            gae_lambda: Lambda para GAE (Generalized Advantage Estimation)
            clip_epsilon: Epsilon para clipping do PPO
            value_coef: Coeficiente da perda do critic
            entropy_coef: Coeficiente do bônus de entropia
            max_grad_norm: Norma máxima para gradient clipping
            hidden_dim: Dimensão das camadas ocultas
            device: Dispositivo (cpu ou cuda)
        """
        self.device = torch.device(device)

        # Hiperparâmetros
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Rede neural
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffer de experiências
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Métricas
        self.episode_rewards = deque(maxlen=100)
        self.training_step = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Seleciona uma ação baseada no estado atual.

        Args:
            state: Estado do ambiente
            deterministic: Se True, usa ação determinística

        Returns:
            Ação a ser executada
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)

        # Armazena para treinamento
        if not deterministic:
            self.states.append(state)
            self.actions.append(action.cpu().numpy()[0])
            self.log_probs.append(float(log_prob.cpu().numpy()))
            self.values.append(float(value.cpu().numpy().item()))

        return action.cpu().numpy()[0]

    def store_transition(self, reward: float, done: bool):
        """
        Armazena uma transição no buffer.

        Args:
            reward: Recompensa recebida
            done: Se o episódio terminou
        """
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula Generalized Advantage Estimation (GAE).

        Args:
            next_value: Valor estimado do próximo estado

        Returns:
            advantages: Vantagens calculadas
            returns: Retornos (targets para o critic)
        """
        values = np.array(self.values + [next_value])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]

        return advantages, returns

    def update(self, next_state: np.ndarray, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Atualiza a política usando PPO.

        Args:
            next_state: Próximo estado (para calcular GAE)
            n_epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch

        Returns:
            Dicionário com métricas de treinamento
        """
        if len(self.states) == 0:
            return {}

        # Calcula valor do próximo estado
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.policy.forward(next_state_tensor)
            next_value = float(next_value.cpu().numpy().item())

        # Calcula advantages e returns
        advantages, returns = self.compute_gae(next_value)

        # Converte para tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normaliza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Treinamento
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(n_epochs):
            # Mini-batch training
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Avalia ações
                log_probs, values, entropy = self.policy.evaluate_actions(batch_states, batch_actions)

                # Ratio para PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Otimização
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Métricas
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # Limpa buffer
        self.clear_buffer()

        # Incrementa contador
        self.training_step += 1

        # Retorna métricas
        metrics = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'training_step': self.training_step
        }

        return metrics

    def clear_buffer(self):
        """Limpa o buffer de experiências."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def save(self, path: str):
        """
        Salva o modelo.

        Args:
            path: Caminho para salvar
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }, path)

    def load(self, path: str):
        """
        Carrega o modelo.

        Args:
            path: Caminho do modelo salvo
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']

    def get_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas do agente.

        Returns:
            Dicionário com métricas
        """
        return {
            'training_step': self.training_step,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'buffer_size': len(self.states)
        }
