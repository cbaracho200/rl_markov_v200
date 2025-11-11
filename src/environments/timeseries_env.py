"""
Ambiente de Reinforcement Learning para otimização de previsão de séries temporais.

Este ambiente permite que um agente RL aprenda a ajustar coeficientes de modelos
supervisionados para maximizar a precisão de previsões de 6 a 12 meses.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeriesEnv(gym.Env):
    """
    Ambiente de RL para otimização de coeficientes de previsão de séries temporais.

    O agente recebe o estado atual da série temporal e deve escolher coeficientes
    ótimos para os modelos de previsão. A recompensa é baseada na precisão da previsão.

    Attributes:
        forecast_horizon: Horizonte de previsão em meses (6 a 12)
        n_coefficients: Número de coeficientes a serem otimizados
        window_size: Tamanho da janela de observação
        data: Série temporal completa
        current_step: Passo atual no episódio
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        forecast_horizon: int = 6,
        window_size: int = 24,
        n_coefficients: int = 10,
        max_steps: int = 100,
        prediction_models: Optional[List] = None
    ):
        """
        Inicializa o ambiente de RL.

        Args:
            data: DataFrame com a série temporal (deve ter coluna 'value')
            forecast_horizon: Meses à frente para previsão (6-12)
            window_size: Tamanho da janela de observação
            n_coefficients: Número de coeficientes para otimizar
            max_steps: Número máximo de passos por episódio
            prediction_models: Lista de modelos supervisionados a serem usados
        """
        super().__init__()

        assert 6 <= forecast_horizon <= 12, "Horizonte de previsão deve estar entre 6 e 12 meses"

        self.data = data.copy()
        self.forecast_horizon = forecast_horizon
        self.window_size = window_size
        self.n_coefficients = n_coefficients
        self.max_steps = max_steps
        self.prediction_models = prediction_models or []

        # Normalização dos dados
        self.scaler = StandardScaler()
        self.data['value_normalized'] = self.scaler.fit_transform(
            self.data[['value']] if 'value' in self.data.columns else self.data.iloc[:, [0]]
        )

        # Espaço de observação: janela de dados + coeficientes atuais + features estatísticas
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size + n_coefficients + 10,),  # +10 para features estatísticas
            dtype=np.float32
        )

        # Espaço de ação: ajustes nos coeficientes (valores contínuos entre -1 e 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_coefficients,),
            dtype=np.float32
        )

        # Estado interno
        self.current_step = 0
        self.current_coefficients = np.ones(n_coefficients) / n_coefficients
        self.episode_predictions = []
        self.episode_rewards = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reseta o ambiente para um novo episódio.

        Returns:
            observation: Estado inicial
            info: Informações adicionais
        """
        super().reset(seed=seed)

        # Reseta estado interno
        self.current_step = 0
        self.current_coefficients = np.ones(self.n_coefficients) / self.n_coefficients
        self.episode_predictions = []
        self.episode_rewards = []

        # Escolhe ponto inicial aleatório (garantindo espaço para previsão)
        max_start = len(self.data) - self.window_size - self.forecast_horizon - 1
        self.start_idx = np.random.randint(0, max_start)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa uma ação no ambiente.

        Args:
            action: Ajustes nos coeficientes (valores entre -1 e 1)

        Returns:
            observation: Novo estado
            reward: Recompensa recebida
            terminated: Se o episódio terminou
            truncated: Se o episódio foi truncado
            info: Informações adicionais
        """
        # Atualiza coeficientes com a ação (com clipping para manter estabilidade)
        self.current_coefficients += action * 0.1  # Learning rate de 0.1
        self.current_coefficients = np.clip(self.current_coefficients, 0.0, 1.0)

        # Normaliza coeficientes para somarem 1 (ensemble weights)
        if self.current_coefficients.sum() > 0:
            self.current_coefficients /= self.current_coefficients.sum()
        else:
            self.current_coefficients = np.ones(self.n_coefficients) / self.n_coefficients

        # Calcula previsão com os coeficientes atuais
        prediction, actual_value = self._make_prediction()

        # Calcula recompensa baseada no erro de previsão
        reward = self._calculate_reward(prediction, actual_value)

        # Armazena resultados
        self.episode_predictions.append(prediction)
        self.episode_rewards.append(reward)

        # Avança para próximo passo
        self.current_step += 1

        # Verifica se episódio terminou
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Observação do novo estado
        observation = self._get_observation()
        info = self._get_info()
        info['prediction'] = prediction
        info['actual'] = actual_value
        info['coefficients'] = self.current_coefficients.copy()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Constrói o vetor de observação do estado atual.

        Returns:
            Vetor de observação combinando dados históricos, coeficientes e features
        """
        # Janela de dados históricos
        window_start = self.start_idx + self.current_step
        window_end = window_start + self.window_size
        window_data = self.data['value_normalized'].iloc[window_start:window_end].values

        # Se não houver dados suficientes, preenche com zeros
        if len(window_data) < self.window_size:
            window_data = np.pad(
                window_data,
                (0, self.window_size - len(window_data)),
                mode='constant'
            )

        # Features estatísticas da janela
        statistical_features = np.array([
            np.mean(window_data),
            np.std(window_data),
            np.min(window_data),
            np.max(window_data),
            np.median(window_data),
            np.percentile(window_data, 25),
            np.percentile(window_data, 75),
            window_data[-1] - window_data[0],  # Tendência
            np.mean(np.diff(window_data)),  # Velocidade média
            np.std(np.diff(window_data))  # Volatilidade
        ])

        # Combina todos os componentes
        observation = np.concatenate([
            window_data,
            self.current_coefficients,
            statistical_features
        ]).astype(np.float32)

        return observation

    def _make_prediction(self) -> Tuple[float, float]:
        """
        Faz uma previsão usando os coeficientes atuais.

        Returns:
            prediction: Valor previsto
            actual_value: Valor real
        """
        # Índice do valor a ser previsto
        prediction_idx = self.start_idx + self.current_step + self.window_size + self.forecast_horizon

        if prediction_idx >= len(self.data):
            # Se não houver dados futuros, usa o último valor disponível
            actual_value = self.data['value_normalized'].iloc[-1]
            prediction = actual_value
            return prediction, actual_value

        actual_value = self.data['value_normalized'].iloc[prediction_idx]

        # Se não houver modelos, usa uma previsão simples baseada na tendência
        if not self.prediction_models or len(self.prediction_models) == 0:
            window_start = self.start_idx + self.current_step
            window_end = window_start + self.window_size
            window_data = self.data['value_normalized'].iloc[window_start:window_end].values

            # Previsão simples: média ponderada de diferentes métodos
            trend = np.polyfit(range(len(window_data)), window_data, 1)[0]
            last_value = window_data[-1]
            mean_value = np.mean(window_data)

            # Usa coeficientes para ponderar diferentes estratégias
            prediction = (
                self.current_coefficients[0] * (last_value + trend * self.forecast_horizon) +  # Tendência linear
                self.current_coefficients[1] * mean_value +  # Média
                self.current_coefficients[2] * last_value +  # Último valor
                np.sum(self.current_coefficients[3:]) * (last_value * 0.95 + mean_value * 0.05)  # Combinação
            )
        else:
            # Usa ensemble de modelos com coeficientes como pesos
            predictions = []
            for i, model in enumerate(self.prediction_models[:self.n_coefficients]):
                if hasattr(model, 'predict'):
                    model_pred = model.predict(self.start_idx + self.current_step)
                    predictions.append(model_pred)

            if predictions:
                # Combina previsões com pesos aprendidos
                prediction = np.average(
                    predictions,
                    weights=self.current_coefficients[:len(predictions)]
                )
            else:
                prediction = actual_value  # Fallback

        return prediction, actual_value

    def _calculate_reward(self, prediction: float, actual_value: float) -> float:
        """
        Calcula a recompensa baseada na precisão da previsão.

        Args:
            prediction: Valor previsto
            actual_value: Valor real

        Returns:
            Recompensa (positiva para previsões precisas)
        """
        # Erro absoluto percentual
        mape = np.abs((actual_value - prediction) / (actual_value + 1e-8)) * 100

        # Erro quadrático médio normalizado
        mse = (actual_value - prediction) ** 2

        # Recompensa baseada em múltiplas métricas
        # Quanto menor o erro, maior a recompensa
        reward = 0.0

        # Componente 1: Recompensa por MAPE baixo
        if mape < 1.0:
            reward += 10.0
        elif mape < 5.0:
            reward += 5.0
        elif mape < 10.0:
            reward += 2.0
        else:
            reward -= mape / 10.0

        # Componente 2: Penalidade por MSE alto
        reward -= mse * 10.0

        # Componente 3: Bônus por consistência (se houver histórico)
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            if reward > avg_reward:
                reward += 1.0  # Bônus por melhorar

        return float(reward)

    def _get_info(self) -> Dict:
        """
        Retorna informações adicionais sobre o estado atual.

        Returns:
            Dicionário com informações
        """
        info = {
            'step': self.current_step,
            'forecast_horizon': self.forecast_horizon,
            'coefficients': self.current_coefficients.copy(),
            'episode_mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'episode_total_reward': np.sum(self.episode_rewards) if self.episode_rewards else 0.0
        }
        return info

    def render(self):
        """Renderiza o estado atual do ambiente."""
        if len(self.episode_rewards) > 0:
            print(f"\n=== Step {self.current_step} ===")
            print(f"Current Coefficients: {self.current_coefficients}")
            print(f"Last Reward: {self.episode_rewards[-1]:.4f}")
            print(f"Mean Episode Reward: {np.mean(self.episode_rewards):.4f}")
            print(f"Total Episode Reward: {np.sum(self.episode_rewards):.4f}")

    def close(self):
        """Limpa recursos do ambiente."""
        pass
