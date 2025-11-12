"""
Framework de Reinforcement Learning para Previsão de Séries Temporais Econômicas

Este framework usa agentes de RL para otimizar coeficientes de modelos supervisionados,
visando previsões precisas com antecedência de 6 a 12 meses.
"""

__version__ = "2.0.0"
__author__ = "Economic Forecasting RL Team"

# Lazy imports para evitar erros quando dependências não estão instaladas
__all__ = []

try:
    from src.environments.timeseries_env import TimeSeriesEnv
    __all__.append('TimeSeriesEnv')
except ImportError:
    pass

try:
    from src.agents.rl_agent import RLAgent
    __all__.append('RLAgent')
except ImportError:
    pass

try:
    from src.agents.rl_agent_advanced import AdvancedRLAgent
    __all__.append('AdvancedRLAgent')
except ImportError:
    pass

try:
    from src.models.ensemble_predictor import EnsemblePredictor
    __all__.append('EnsemblePredictor')
except ImportError:
    pass
