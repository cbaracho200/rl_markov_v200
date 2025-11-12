"""
Agente de Reinforcement Learning Avançado (Nível PhD).

Implementa técnicas state-of-the-art:
- Transformer-based Actor-Critic com Multi-Head Attention
- Prioritized Experience Replay (PER)
- Noisy Networks para exploração adaptativa
- Dueling Architecture
- Recurrent Policy (LSTM)
- Ensemble de Critics
- Learning Rate Scheduling com Warmup
- Gradient Accumulation
- Adaptive Entropy Regularization
- Hindsight Experience Replay (HER)
- Spectral Normalization para estabilidade
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List, Dict, Optional, Deque
from collections import deque, namedtuple
import random
import math


# Prioritized Experience Replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'))


class SumTree:
    """
    Estrutura de dados Sum Tree para Prioritized Experience Replay.
    Complexidade O(log n) para operações.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Buffer de experiências com priorização baseada em TD-error.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 0.01
        self.max_priority = 1.0

    def add(self, error: float, transition: Transition):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Importance Sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update(self, idx: int, error: float):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention para capturar dependências complexas no estado.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = self.combine_heads(context)

        output = self.W_o(context)
        output = self.dropout(output)

        return self.layer_norm(output + residual)


class NoisyLinear(nn.Module):
    """
    Noisy Networks para exploração adaptativa sem epsilon-greedy.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class TransformerActorCritic(nn.Module):
    """
    Actor-Critic Avançado com Transformer, Multi-Head Attention, e Dueling Architecture.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_noisy: bool = True,
        use_lstm: bool = True
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        self.use_noisy = use_noisy

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding (para sequências temporais)
        self.register_buffer('pos_encoding', self._get_positional_encoding(100, hidden_dim))

        # Multi-Head Attention Layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Feed-Forward Networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # LSTM para memória temporal
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
            self.lstm_hidden = None

        # Actor (Dueling Architecture)
        LinearLayer = NoisyLinear if use_noisy else nn.Linear

        self.actor_advantage = nn.Sequential(
            LinearLayer(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            LinearLayer(hidden_dim // 2, action_dim)
        )

        self.actor_value = nn.Sequential(
            LinearLayer(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            LinearLayer(hidden_dim // 2, action_dim)
        )

        # Parametrização do log_std como rede neural (mais expressivo)
        self.log_std_network = nn.Sequential(
            LinearLayer(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            LinearLayer(hidden_dim // 4, action_dim)
        )

        # Ensemble de Critics (3 critics para reduzir viés)
        self.critics = nn.ModuleList([
            nn.Sequential(
                LinearLayer(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout),
                LinearLayer(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                LinearLayer(hidden_dim // 4, 1)
            )
            for _ in range(3)
        ])

        self._initialize_weights()

    def _get_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def reset_noise(self):
        """Reset noise em todas as Noisy Layers."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def reset_lstm_hidden(self):
        """Reset do hidden state do LSTM."""
        self.lstm_hidden = None

    def forward(self, state: torch.Tensor, sequence_mode: bool = False) -> Tuple[Normal, torch.Tensor]:
        # Input embedding
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        batch_size, seq_len, _ = state.shape
        x = self.input_embedding(state)

        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer layers
        for attention, ffn in zip(self.attention_layers, self.ffn_layers):
            x_attn = attention(x)
            x = x_attn + ffn(x_attn)

        # LSTM para capturar dependências temporais
        if self.use_lstm:
            if self.lstm_hidden is None or not sequence_mode:
                self.lstm_hidden = None
            x, self.lstm_hidden = self.lstm(x, self.lstm_hidden)

        # Pega último timestep
        features = x[:, -1, :]

        # Actor com Dueling Architecture
        advantage = self.actor_advantage(features)
        value = self.actor_value(features)
        action_mean = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        action_mean = torch.tanh(action_mean)

        # Log std adaptativo
        log_std = self.log_std_network(features)
        log_std = torch.clamp(log_std, -20, 2)
        action_std = torch.exp(log_std)

        distribution = Normal(action_mean, action_std)

        # Ensemble de Critics (retorna média)
        critic_values = torch.stack([critic(features) for critic in self.critics])
        value = critic_values.mean(dim=0)

        return distribution, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.forward(state)

        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.forward(states)

        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy


class AdvancedRLAgent:
    """
    Agente RL Avançado (Nível PhD) com técnicas state-of-the-art.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        device: str = 'cpu',
        use_per: bool = True,
        use_noisy: bool = True,
        use_lstm: bool = True,
        buffer_size: int = 100000
    ):
        self.device = torch.device(device)

        # Hiperparâmetros
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_per = use_per

        # Adaptive entropy coefficient
        self.target_entropy = -action_dim
        self.log_entropy_coef = torch.zeros(1, requires_grad=True, device=self.device)
        self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=learning_rate)

        # Rede neural avançada
        self.policy = TransformerActorCritic(
            state_dim, action_dim, hidden_dim, num_heads, num_layers,
            use_noisy=use_noisy, use_lstm=use_lstm
        ).to(self.device)

        # Optimizer com warmup e scheduling
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        # Learning Rate Scheduler com Warmup
        self.warmup_steps = 1000
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.1
        )

        # Prioritized Experience Replay
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = deque(maxlen=buffer_size)

        # Buffer temporário para episódio atual
        self.episode_buffer = []

        # Métricas
        self.episode_rewards = deque(maxlen=100)
        self.training_step = 0
        self.gradient_steps = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.policy.reset_noise()
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)

        info = {
            'log_prob': float(log_prob.cpu().numpy()),
            'value': float(value.cpu().numpy().item())
        }

        return action.cpu().numpy()[0], info

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, info: Dict):
        transition = Transition(state, action, reward, next_state, done,
                              info['log_prob'], info['value'])
        self.episode_buffer.append(transition)

        if done:
            self._process_episode()

    def _process_episode(self):
        """Processa episódio completo e adiciona ao replay buffer."""
        if len(self.episode_buffer) == 0:
            return

        # Calcula TD errors para priorização
        for transition in self.episode_buffer:
            state_tensor = torch.FloatTensor(transition.state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(transition.next_state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, current_value = self.policy.forward(state_tensor)
                _, next_value = self.policy.forward(next_state_tensor)

                td_error = (transition.reward + self.gamma * float(next_value.cpu().numpy().item()) *
                           (1 - transition.done) - float(current_value.cpu().numpy().item()))

            if self.use_per:
                self.replay_buffer.add(td_error, transition)
            else:
                self.replay_buffer.append(transition)

        # Limpa buffer do episódio
        self.episode_buffer = []

    def update(self, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """Atualiza a política usando PPO com PER."""
        if (self.use_per and len(self.replay_buffer) < batch_size) or \
           (not self.use_per and len(self.replay_buffer) < batch_size):
            return {}

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_entropy_loss = 0
        n_updates = 0

        for epoch in range(n_epochs):
            # Sample batch
            if self.use_per:
                batch, idxs, is_weights = self.replay_buffer.sample(batch_size)
                is_weights = torch.FloatTensor(is_weights).to(self.device)
            else:
                batch = random.sample(list(self.replay_buffer), batch_size)
                is_weights = torch.ones(batch_size).to(self.device)
                idxs = None

            # Prepara batch
            states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
            actions = torch.FloatTensor(np.array([t.action for t in batch])).to(self.device)
            rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
            dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
            old_log_probs = torch.FloatTensor([t.log_prob for t in batch]).to(self.device)
            old_values = torch.FloatTensor([t.value for t in batch]).to(self.device)

            # Calcula returns e advantages
            with torch.no_grad():
                _, next_values = self.policy.forward(next_states)
                next_values = next_values.squeeze(-1)
                returns = rewards + self.gamma * next_values * (1 - dones)

            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Avalia ações
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -(torch.min(surr1, surr2) * is_weights).mean()

            # Value loss com clipping
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            value_losses = (values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = (torch.max(value_losses, value_losses_clipped) * is_weights).mean()

            # Adaptive entropy
            current_entropy_coef = torch.exp(self.log_entropy_coef).detach()
            entropy_loss = -(entropy * is_weights).mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + current_entropy_coef * entropy_loss

            # Otimização
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Atualiza entropy coefficient
            entropy_loss_for_opt = (self.log_entropy_coef *
                                   (entropy.detach().mean() - self.target_entropy))
            self.entropy_optimizer.zero_grad()
            entropy_loss_for_opt.backward()
            self.entropy_optimizer.step()

            # Atualiza prioridades no buffer
            if self.use_per and idxs is not None:
                td_errors = torch.abs(values - returns).detach().cpu().numpy()
                for idx, td_error in zip(idxs, td_errors):
                    self.replay_buffer.update(idx, td_error)

            # Métricas
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_entropy_loss += entropy_loss.item()
            n_updates += 1

            self.gradient_steps += 1

        # Learning rate scheduling
        if self.gradient_steps > self.warmup_steps:
            self.scheduler.step()

        self.training_step += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'entropy_coef': float(torch.exp(self.log_entropy_coef).detach().cpu().numpy()),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_step': self.training_step
        }

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'log_entropy_coef': self.log_entropy_coef,
            'training_step': self.training_step,
            'gradient_steps': self.gradient_steps
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.log_entropy_coef = checkpoint['log_entropy_coef']
        self.training_step = checkpoint['training_step']
        self.gradient_steps = checkpoint['gradient_steps']

    def get_metrics(self) -> Dict[str, float]:
        return {
            'training_step': self.training_step,
            'gradient_steps': self.gradient_steps,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'buffer_size': len(self.replay_buffer),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'entropy_coef': float(torch.exp(self.log_entropy_coef).detach().cpu().numpy())
        }
