"""
Métricas de avaliação para previsões de séries temporais.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de avaliação completas.

    Args:
        y_true: Valores reais
        y_pred: Valores previstos

    Returns:
        Dicionário com métricas
    """
    # Garante arrays numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Métricas básicas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Direção da previsão (acurácia direcional)
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0.0

    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'smape': float(smape),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy),
        'max_error': float(max_error)
    }


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] = [1, 3, 6, 12]
) -> Dict[int, Dict[str, float]]:
    """
    Calcula métricas para diferentes horizontes de previsão.

    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        horizons: Lista de horizontes para avaliar

    Returns:
        Dicionário com métricas por horizonte
    """
    results = {}

    for h in horizons:
        if h <= len(y_true) and h <= len(y_pred):
            metrics = calculate_metrics(y_true[:h], y_pred[:h])
            results[h] = metrics

    return results


def compare_models(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    metric: str = 'mape'
) -> pd.DataFrame:
    """
    Compara múltiplos modelos.

    Args:
        y_true: Valores reais
        predictions_dict: Dicionário {nome_modelo: previsões}
        metric: Métrica principal para ranking

    Returns:
        DataFrame com comparação
    """
    results = []

    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_pred)
        metrics['model'] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)

    # Reordena colunas
    cols = ['model'] + [c for c in df.columns if c != 'model']
    df = df[cols]

    # Ordena por métrica (menor é melhor para maioria das métricas)
    if metric in ['r2', 'directional_accuracy']:
        df = df.sort_values(metric, ascending=False)
    else:
        df = df.sort_values(metric, ascending=True)

    return df


def rolling_forecast_validation(
    data: pd.Series,
    model,
    initial_window: int,
    horizon: int,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Validação com janela deslizante (rolling forecast).

    Args:
        data: Série temporal completa
        model: Modelo com métodos fit() e predict()
        initial_window: Tamanho da janela inicial de treino
        horizon: Horizonte de previsão
        step: Passo da janela deslizante

    Returns:
        y_true, y_pred, metrics_list
    """
    y_true_list = []
    y_pred_list = []
    metrics_list = []

    n = len(data)

    for i in range(initial_window, n - horizon, step):
        # Dados de treino
        train_data = data[:i]

        # Dados de teste
        test_data = data[i:i + horizon]

        # Treina modelo
        model.fit(train_data)

        # Faz previsão
        predictions = model.predict(steps=horizon)

        # Limita ao tamanho real
        predictions = predictions[:len(test_data)]

        # Armazena
        y_true_list.extend(test_data.values)
        y_pred_list.extend(predictions)

        # Calcula métricas para esta janela
        window_metrics = calculate_metrics(test_data.values, predictions)
        window_metrics['window_start'] = i
        metrics_list.append(window_metrics)

    return np.array(y_true_list), np.array(y_pred_list), metrics_list


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calcula Sharpe Ratio para estratégia baseada em previsões.

    Args:
        returns: Array de retornos
        risk_free_rate: Taxa livre de risco

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if np.std(returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(returns)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calcula o máximo drawdown.

    Args:
        cumulative_returns: Retornos cumulativos

    Returns:
        Máximo drawdown (valor negativo)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
    return float(np.min(drawdown))
