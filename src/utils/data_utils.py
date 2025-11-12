"""
Utilitários para manipulação de dados de séries temporais.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta


def generate_synthetic_data(
    n_points: int = 500,
    trend: float = 0.02,
    seasonality_amplitude: float = 10.0,
    seasonality_period: int = 12,
    noise_std: float = 2.0,
    start_value: float = 100.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Gera dados sintéticos de séries temporais econômicas.

    Args:
        n_points: Número de pontos
        trend: Tendência linear
        seasonality_amplitude: Amplitude da sazonalidade
        seasonality_period: Período da sazonalidade
        noise_std: Desvio padrão do ruído
        start_value: Valor inicial
        seed: Seed para reprodutibilidade

    Returns:
        DataFrame com a série temporal
    """
    if seed is not None:
        np.random.seed(seed)

    # Componente de tendência
    time = np.arange(n_points)
    trend_component = start_value + trend * time

    # Componente sazonal
    seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * time / seasonality_period)

    # Componente cíclica (ciclo econômico)
    cycle_period = 48  # Ciclo de 4 anos
    cycle_component = 5 * np.sin(2 * np.pi * time / cycle_period + np.pi / 4)

    # Ruído
    noise = np.random.normal(0, noise_std, n_points)

    # Componente autorregressivo
    ar_component = np.zeros(n_points)
    ar_component[0] = 0
    for i in range(1, n_points):
        ar_component[i] = 0.7 * ar_component[i-1] + np.random.normal(0, noise_std * 0.5)

    # Série temporal completa
    values = trend_component + seasonal_component + cycle_component + noise + ar_component

    # Cria DataFrame
    start_date = datetime(2010, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(n_points)]

    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend_component,
        'seasonal': seasonal_component,
        'cycle': cycle_component,
        'noise': noise
    })

    return df


def load_economic_data(filepath: str, value_column: str = 'value', date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega dados econômicos de um arquivo.

    Args:
        filepath: Caminho do arquivo (CSV, Excel, etc.)
        value_column: Nome da coluna com valores
        date_column: Nome da coluna com datas (opcional)

    Returns:
        DataFrame com os dados
    """
    # Detecta extensão
    ext = filepath.split('.')[-1].lower()

    if ext == 'csv':
        df = pd.read_csv(filepath)
    elif ext in ['xlsx', 'xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")

    # Processa coluna de data
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

    # Renomeia coluna de valor se necessário
    if value_column != 'value' and value_column in df.columns:
        df['value'] = df[value_column]

    return df


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide dados em conjuntos de treino, validação e teste.

    Args:
        data: DataFrame com os dados
        train_ratio: Proporção de dados para treino
        val_ratio: Proporção de dados para validação
        test_ratio: Proporção de dados para teste

    Returns:
        train_data, val_data, test_data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "As proporções devem somar 1.0"

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()

    return train_data, val_data, test_data


def create_lagged_features(
    data: pd.Series,
    lags: list = [1, 2, 3, 6, 12],
    rolling_windows: list = [3, 6, 12]
) -> pd.DataFrame:
    """
    Cria features de lag e rolling statistics.

    Args:
        data: Série temporal
        lags: Lista de lags a criar
        rolling_windows: Janelas para médias móveis

    Returns:
        DataFrame com features
    """
    df = pd.DataFrame({'value': data})

    # Lags
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window).max()

    # Remove NaN
    df = df.dropna()

    return df


def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
    """
    Normaliza dados.

    Args:
        data: DataFrame com dados
        method: Método de normalização ('minmax' ou 'standard')

    Returns:
        DataFrame normalizado e parâmetros de normalização
    """
    df = data.copy()
    params = {}

    for col in df.columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
            params[col] = {'min': min_val, 'max': max_val}

        elif method == 'standard':
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[col] = (df[col] - mean_val) / (std_val + 1e-8)
            params[col] = {'mean': mean_val, 'std': std_val}

    return df, params


def denormalize_data(data: pd.DataFrame, params: Dict, method: str = 'minmax') -> pd.DataFrame:
    """
    Desnormaliza dados.

    Args:
        data: DataFrame normalizado
        params: Parâmetros de normalização
        method: Método usado na normalização

    Returns:
        DataFrame desnormalizado
    """
    df = data.copy()

    for col in df.columns:
        if col in params:
            if method == 'minmax':
                min_val = params[col]['min']
                max_val = params[col]['max']
                df[col] = df[col] * (max_val - min_val) + min_val

            elif method == 'standard':
                mean_val = params[col]['mean']
                std_val = params[col]['std']
                df[col] = df[col] * std_val + mean_val

    return df
