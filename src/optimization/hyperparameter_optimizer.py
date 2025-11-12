"""
Sistema de Otimização Recursiva de Hiperparâmetros usando Optuna.

Otimiza automaticamente hiperparâmetros dos modelos durante o treinamento.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """
    Otimizador de hiperparâmetros usando Optuna (Bayesian Optimization).

    Características:
    - Otimização bayesiana (mais eficiente que grid search)
    - Pruning automático de trials ruins
    - Suporte a múltiplos modelos simultaneamente
    - Otimização recursiva durante treinamento
    - Histórico completo de trials
    """

    def __init__(
        self,
        metric: str = 'mape',
        direction: str = 'minimize',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True
    ):
        """
        Inicializa o otimizador.

        Args:
            metric: Métrica a otimizar ('mape', 'rmse', 'mae', etc)
            direction: 'minimize' ou 'maximize'
            n_trials: Número de trials para otimização
            timeout: Tempo máximo em segundos (None = sem limite)
            n_jobs: Número de jobs paralelos
            verbose: Se True, mostra progresso
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna não instalado. Instale com: pip install optuna")

        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.study = None
        self.best_params = {}
        self.optimization_history = []

    def optimize_model(
        self,
        model_class: Any,
        train_data: Union[np.ndarray, pd.Series],
        val_data: Union[np.ndarray, pd.Series],
        param_space: Dict[str, tuple],
        forecast_horizon: int = 12
    ) -> Dict[str, Any]:
        """
        Otimiza hiperparâmetros de um modelo.

        Args:
            model_class: Classe do modelo a otimizar
            train_data: Dados de treino
            val_data: Dados de validação
            param_space: Espaço de busca dos parâmetros
            forecast_horizon: Horizonte de previsão

        Returns:
            Melhores parâmetros encontrados
        """
        import optuna

        def objective(trial):
            """Função objetivo para Optuna."""
            # Sugere hiperparâmetros
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config[0]

                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config[1],
                        param_config[2]
                    )
                elif param_type == 'float':
                    if len(param_config) > 3 and param_config[3] == 'log':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config[1],
                            param_config[2],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config[1],
                            param_config[2]
                        )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config[1]
                    )

            try:
                # Cria e treina modelo
                model = model_class(**params)
                model.fit(train_data)

                # Faz previsão
                predictions = model.predict(steps=min(forecast_horizon, len(val_data)))
                actual = val_data[:len(predictions)]

                # Calcula métrica
                if self.metric == 'mape':
                    score = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
                elif self.metric == 'rmse':
                    score = np.sqrt(np.mean((actual - predictions) ** 2))
                elif self.metric == 'mae':
                    score = np.mean(np.abs(actual - predictions))
                elif self.metric == 'r2':
                    ss_res = np.sum((actual - predictions) ** 2)
                    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                    score = 1 - (ss_res / ss_tot)
                else:
                    raise ValueError(f"Métrica desconhecida: {self.metric}")

                return score

            except Exception as e:
                if self.verbose:
                    print(f"  Trial falhou: {e}")
                # Retorna valor ruim para pruning
                return float('inf') if self.direction == 'minimize' else float('-inf')

        # Cria study
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=f"optimize_{model_class.__name__}"
        )

        # Otimiza
        if self.verbose:
            print(f"\n🔍 Otimizando {model_class.__name__}...")
            print(f"   Trials: {self.n_trials}")
            print(f"   Métrica: {self.metric} ({self.direction})")

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose
        )

        # Salva melhores parâmetros
        self.best_params[model_class.__name__] = self.study.best_params

        if self.verbose:
            print(f"\n✅ Otimização concluída!")
            print(f"   Melhor {self.metric}: {self.study.best_value:.4f}")
            print(f"   Melhores parâmetros:")
            for param, value in self.study.best_params.items():
                print(f"      {param}: {value}")

        # Salva histórico
        self.optimization_history.append({
            'model': model_class.__name__,
            'best_params': self.study.best_params.copy(),
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials)
        })

        return self.study.best_params

    def optimize_ensemble(
        self,
        model_configs: List[Dict],
        train_data: Union[np.ndarray, pd.Series],
        val_data: Union[np.ndarray, pd.Series],
        forecast_horizon: int = 12
    ) -> Dict[str, Dict[str, Any]]:
        """
        Otimiza todos os modelos do ensemble.

        Args:
            model_configs: Lista de dicts com 'class', 'param_space'
            train_data: Dados de treino
            val_data: Dados de validação
            forecast_horizon: Horizonte de previsão

        Returns:
            Dict com melhores parâmetros para cada modelo
        """
        all_best_params = {}

        for config in model_configs:
            model_class = config['class']
            param_space = config['param_space']

            best_params = self.optimize_model(
                model_class=model_class,
                train_data=train_data,
                val_data=val_data,
                param_space=param_space,
                forecast_horizon=forecast_horizon
            )

            all_best_params[model_class.__name__] = best_params

        return all_best_params

    def get_optimization_history(self) -> List[Dict]:
        """
        Retorna histórico completo de otimizações.

        Returns:
            Lista com histórico
        """
        return self.optimization_history

    def plot_optimization_history(self, model_name: Optional[str] = None):
        """
        Plota histórico de otimização.

        Args:
            model_name: Nome do modelo (None = último)
        """
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            import matplotlib.pyplot as plt

            if self.study is None:
                print("⚠️  Nenhuma otimização realizada ainda.")
                return

            # Plot 1: Histórico
            fig1 = plot_optimization_history(self.study)
            fig1.show()

            # Plot 2: Importância dos parâmetros
            try:
                fig2 = plot_param_importances(self.study)
                fig2.show()
            except:
                pass  # Pode falhar se poucos trials

        except ImportError:
            print("⚠️  Plotly não instalado. Instale com: pip install plotly")


class RecursiveOptimizer:
    """
    Otimizador recursivo que ajusta hiperparâmetros durante o treinamento.

    A cada N episódios, reotimiza os hiperparâmetros baseado no desempenho recente.
    """

    def __init__(
        self,
        hyperparameter_optimizer: HyperparameterOptimizer,
        reoptimize_frequency: int = 50,
        performance_window: int = 20,
        improvement_threshold: float = 0.05
    ):
        """
        Inicializa o otimizador recursivo.

        Args:
            hyperparameter_optimizer: Otimizador de hiperparâmetros
            reoptimize_frequency: Reotimiza a cada N episódios
            performance_window: Janela para calcular performance
            improvement_threshold: Threshold de melhoria para reotimizar
        """
        self.hp_optimizer = hyperparameter_optimizer
        self.reoptimize_frequency = reoptimize_frequency
        self.performance_window = performance_window
        self.improvement_threshold = improvement_threshold

        self.episode_count = 0
        self.performance_history = []
        self.reoptimization_history = []

    def should_reoptimize(self, current_performance: float) -> bool:
        """
        Decide se deve reotimizar baseado no desempenho recente.

        Args:
            current_performance: Performance atual

        Returns:
            True se deve reotimizar
        """
        self.episode_count += 1
        self.performance_history.append(current_performance)

        # Verifica frequência
        if self.episode_count % self.reoptimize_frequency != 0:
            return False

        # Verifica se há histórico suficiente
        if len(self.performance_history) < self.performance_window * 2:
            return False

        # Compara performance recente com anterior
        recent_perf = np.mean(self.performance_history[-self.performance_window:])
        previous_perf = np.mean(
            self.performance_history[-2*self.performance_window:-self.performance_window]
        )

        # Calcula melhoria
        if self.hp_optimizer.direction == 'minimize':
            improvement = (previous_perf - recent_perf) / previous_perf
        else:
            improvement = (recent_perf - previous_perf) / previous_perf

        # Reotimiza se não melhorou o suficiente
        should_reopt = improvement < self.improvement_threshold

        if should_reopt:
            self.reoptimization_history.append({
                'episode': self.episode_count,
                'recent_performance': recent_perf,
                'previous_performance': previous_perf,
                'improvement': improvement
            })

        return should_reopt

    def reoptimize(
        self,
        model_configs: List[Dict],
        train_data: Union[np.ndarray, pd.Series],
        val_data: Union[np.ndarray, pd.Series],
        forecast_horizon: int = 12
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executa reotimização.

        Args:
            model_configs: Configurações dos modelos
            train_data: Dados de treino
            val_data: Dados de validação
            forecast_horizon: Horizonte

        Returns:
            Novos melhores parâmetros
        """
        print(f"\n🔄 Reotimização recursiva no episódio {self.episode_count}")

        new_params = self.hp_optimizer.optimize_ensemble(
            model_configs=model_configs,
            train_data=train_data,
            val_data=val_data,
            forecast_horizon=forecast_horizon
        )

        return new_params

    def get_reoptimization_history(self) -> List[Dict]:
        """Retorna histórico de reotimizações."""
        return self.reoptimization_history
