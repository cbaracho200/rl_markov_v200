"""
Funções de visualização para análise de resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = "Previsões vs Valores Reais",
    save_path: Optional[str] = None
):
    """
    Plota previsões vs valores reais.

    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        dates: Datas (opcional)
        title: Título do gráfico
        save_path: Caminho para salvar (opcional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = dates if dates is not None else np.arange(len(y_true))

    # Plot principal
    ax1.plot(x, y_true, label='Real', linewidth=2, marker='o', markersize=4)
    ax1.plot(x, y_pred, label='Previsto', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax1.fill_between(x, y_true, y_pred, alpha=0.2)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Valor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot de erros
    errors = y_true - y_pred
    ax2.bar(x, errors, alpha=0.6, color='red', label='Erro')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Erros de Previsão', fontsize=12)
    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Erro')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.show()


def plot_coefficients(
    coefficients: np.ndarray,
    model_names: Optional[List[str]] = None,
    title: str = "Coeficientes do Ensemble",
    save_path: Optional[str] = None
):
    """
    Plota coeficientes do ensemble.

    Args:
        coefficients: Array de coeficientes
        model_names: Nomes dos modelos (opcional)
        title: Título do gráfico
        save_path: Caminho para salvar (opcional)
    """
    if model_names is None:
        model_names = [f'Modelo {i+1}' for i in range(len(coefficients))]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(coefficients)))
    bars = ax.bar(model_names, coefficients, color=colors, alpha=0.7, edgecolor='black')

    # Adiciona valores nas barras
    for bar, coef in zip(bars, coefficients):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{coef:.3f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Peso', fontsize=12)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.show()


def plot_backtest_results(
    results: Dict,
    title: str = "Resultados do Backtesting",
    save_path: Optional[str] = None
):
    """
    Plota resultados detalhados do backtesting.

    Args:
        results: Dicionário com resultados
        title: Título do gráfico
        save_path: Caminho para salvar (opcional)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Previsões vs Real
    ax1 = fig.add_subplot(gs[0, :])
    if 'y_true' in results and 'y_pred' in results:
        y_true = results['y_true']
        y_pred = results['y_pred']
        x = np.arange(len(y_true))

        ax1.plot(x, y_true, label='Real', linewidth=2, color='blue')
        ax1.plot(x, y_pred, label='Previsto', linewidth=2, color='red', alpha=0.7)
        ax1.fill_between(x, y_true, y_pred, alpha=0.2)
        ax1.set_title('Previsões vs Valores Reais', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Período')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Distribuição dos erros
    ax2 = fig.add_subplot(gs[1, 0])
    if 'y_true' in results and 'y_pred' in results:
        errors = results['y_true'] - results['y_pred']
        ax2.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Distribuição dos Erros', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Erro')
        ax2.set_ylabel('Frequência')
        ax2.grid(True, alpha=0.3)

    # 3. Scatter plot
    ax3 = fig.add_subplot(gs[1, 1])
    if 'y_true' in results and 'y_pred' in results:
        ax3.scatter(results['y_true'], results['y_pred'], alpha=0.5, s=50)

        # Linha de perfeição
        min_val = min(results['y_true'].min(), results['y_pred'].min())
        max_val = max(results['y_true'].max(), results['y_pred'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')

        ax3.set_title('Real vs Previsto', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Valor Real')
        ax3.set_ylabel('Valor Previsto')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Métricas
    ax4 = fig.add_subplot(gs[2, 0])
    if 'metrics' in results:
        metrics = results['metrics']
        metric_names = list(metrics.keys())[:6]  # Primeiras 6 métricas
        metric_values = [metrics[m] for m in metric_names]

        colors = ['green' if 'r2' in m or 'directional' in m else 'orange' for m in metric_names]
        bars = ax4.barh(metric_names, metric_values, color=colors, alpha=0.7)

        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax4.text(val, i, f' {val:.3f}', va='center', fontweight='bold')

        ax4.set_title('Métricas de Avaliação', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Valor')
        ax4.grid(True, alpha=0.3, axis='x')

    # 5. Evolução temporal do erro
    ax5 = fig.add_subplot(gs[2, 1])
    if 'y_true' in results and 'y_pred' in results:
        errors = np.abs(results['y_true'] - results['y_pred'])
        x = np.arange(len(errors))

        ax5.plot(x, errors, linewidth=2, color='red', alpha=0.7)
        ax5.fill_between(x, 0, errors, alpha=0.3, color='red')
        ax5.axhline(y=np.mean(errors), color='blue', linestyle='--',
                    linewidth=2, label=f'Erro Médio: {np.mean(errors):.3f}')
        ax5.set_title('Evolução do Erro Absoluto', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Período')
        ax5.set_ylabel('Erro Absoluto')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.show()


def plot_multiple_forecasts(
    y_true: np.ndarray,
    forecasts_dict: Dict[str, np.ndarray],
    title: str = "Comparação de Modelos",
    save_path: Optional[str] = None
):
    """
    Plota previsões de múltiplos modelos.

    Args:
        y_true: Valores reais
        forecasts_dict: Dicionário {nome_modelo: previsões}
        title: Título do gráfico
        save_path: Caminho para salvar (opcional)
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(y_true))

    # Plot valores reais
    ax.plot(x, y_true, label='Real', linewidth=3, color='black', marker='o', markersize=6)

    # Plot previsões de cada modelo
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts_dict)))

    for (model_name, forecast), color in zip(forecasts_dict.items(), colors):
        ax.plot(x, forecast, label=model_name, linewidth=2, alpha=0.7,
                color=color, marker='s', markersize=4)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Período', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.show()


def plot_rl_training_history(
    history: Dict,
    save_path: Optional[str] = None
):
    """
    Plota histórico de treinamento do RL.

    Args:
        history: Dicionário com histórico
        save_path: Caminho para salvar (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Histórico de Treinamento RL', fontsize=16, fontweight='bold')

    # Recompensas
    if 'episode_rewards' in history:
        axes[0, 0].plot(history['episode_rewards'], alpha=0.6, label='Por episódio')
        if 'mean_rewards' in history and len(history['mean_rewards']) > 0:
            axes[0, 0].plot(history['mean_rewards'], linewidth=2, label='Média móvel', color='red')
        axes[0, 0].set_title('Recompensas')
        axes[0, 0].set_xlabel('Episódio')
        axes[0, 0].set_ylabel('Recompensa')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Perdas
    if 'policy_losses' in history and len(history['policy_losses']) > 0:
        axes[0, 1].plot(history['policy_losses'], label='Policy Loss', alpha=0.7)
        if 'value_losses' in history:
            axes[0, 1].plot(history['value_losses'], label='Value Loss', alpha=0.7)
        axes[0, 1].set_title('Perdas')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Comprimento dos episódios
    if 'episode_lengths' in history:
        axes[1, 0].plot(history['episode_lengths'], color='green', alpha=0.7)
        axes[1, 0].set_title('Comprimento dos Episódios')
        axes[1, 0].set_xlabel('Episódio')
        axes[1, 0].set_ylabel('Passos')
        axes[1, 0].grid(True, alpha=0.3)

    # Entropia
    if 'entropies' in history and len(history['entropies']) > 0:
        axes[1, 1].plot(history['entropies'], color='purple', alpha=0.7)
        axes[1, 1].set_title('Entropia da Política')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Entropia')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")

    plt.show()
