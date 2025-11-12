"""
Framework de Reinforcement Learning para Previsão de Séries Temporais Econômicas

Este framework usa agentes de RL para otimizar coeficientes de modelos supervisionados,
visando previsões precisas com antecedência de 6 a 12 meses.
"""

__version__ = "1.0.0"
__author__ = "Economic Forecasting RL Team"

from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.models.ensemble_predictor import EnsemblePredictor

__all__ = ['TimeSeriesEnv', 'RLAgent', 'EnsemblePredictor']
