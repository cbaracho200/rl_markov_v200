"""
Módulo de Validação de Variáveis Preditoras
==========================================

Este módulo fornece ferramentas avançadas para validação estatística
de variáveis preditoras em séries temporais, incluindo:

- Testes de estacionaridade (ADF, KPSS, Phillips-Perron)
- Testes de causalidade de Granger
- Seleção automática de variáveis preditoras
- Pipeline integrado de validação

Autor: Advanced RL Framework
Nível: PhD
"""

from .stationarity_tests import StationarityTests, make_stationary
from .granger_causality import GrangerCausality
from .variable_validator import VariableValidator

__all__ = [
    'StationarityTests',
    'make_stationary',
    'GrangerCausality',
    'VariableValidator'
]
