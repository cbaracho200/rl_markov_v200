"""
Ensemble de modelos para previsão otimizada por RL.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional
from .base_model import BasePredictor


class EnsemblePredictor:
    """
    Ensemble de modelos supervisionados com pesos otimizados por RL.

    Combina múltiplos modelos (ARIMA, LSTM, XGBoost) usando pesos
    aprendidos pelo agente de RL para maximizar precisão.
    """

    def __init__(self, models: List[BasePredictor], weights: Optional[np.ndarray] = None):
        """
        Inicializa o ensemble.

        Args:
            models: Lista de modelos base
            weights: Pesos para cada modelo (se None, usa pesos iguais)
        """
        self.models = models
        self.n_models = len(models)

        if weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            assert len(weights) == self.n_models, "Número de pesos deve ser igual ao número de modelos"
            self.weights = weights / weights.sum()  # Normaliza

        self.is_fitted = False

    def fit(self, data: Union[np.ndarray, pd.Series], **kwargs):
        """
        Treina todos os modelos do ensemble.

        Args:
            data: Dados de treinamento
            **kwargs: Argumentos para os modelos
        """
        print(f"\nTreinando ensemble com {self.n_models} modelos...")

        for i, model in enumerate(self.models):
            print(f"\n[{i+1}/{self.n_models}] Treinando {model.name}...")
            try:
                model.fit(data, **kwargs)
                if model.is_fitted:
                    print(f"✓ {model.name} treinado com sucesso")
                else:
                    print(f"✗ {model.name} falhou no treinamento")
            except Exception as e:
                print(f"✗ Erro ao treinar {model.name}: {e}")

        # Verifica se pelo menos um modelo foi treinado
        fitted_models = [m for m in self.models if m.is_fitted]
        if len(fitted_models) > 0:
            self.is_fitted = True
            print(f"\n✓ Ensemble treinado: {len(fitted_models)}/{self.n_models} modelos")
        else:
            self.is_fitted = False
            print("\n✗ Nenhum modelo foi treinado com sucesso")

    def predict(self, steps: int = 1, return_individual: bool = False) -> Union[np.ndarray, Dict]:
        """
        Faz previsões usando ensemble ponderado.

        Args:
            steps: Número de passos à frente
            return_individual: Se True, retorna também previsões individuais

        Returns:
            Previsões do ensemble (ou dicionário com todas as previsões)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble não foi treinado. Chame fit() primeiro.")

        predictions = []
        model_predictions = {}

        for i, model in enumerate(self.models):
            if model.is_fitted:
                try:
                    pred = model.predict(steps=steps)
                    predictions.append(pred)
                    model_predictions[model.name] = pred
                except Exception as e:
                    print(f"Erro ao prever com {model.name}: {e}")
                    # Usa zeros como fallback
                    predictions.append(np.zeros(steps))
                    model_predictions[model.name] = np.zeros(steps)
            else:
                predictions.append(np.zeros(steps))
                model_predictions[model.name] = np.zeros(steps)

        # Combina previsões com pesos
        predictions = np.array(predictions)
        ensemble_prediction = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])

        if return_individual:
            return {
                'ensemble': ensemble_prediction,
                'individual': model_predictions,
                'weights': self.weights.copy()
            }
        else:
            return ensemble_prediction

    def forecast(self, data: Union[np.ndarray, pd.Series], horizon: int) -> np.ndarray:
        """
        Treina e faz previsão.

        Args:
            data: Dados históricos
            horizon: Passos à frente

        Returns:
            Previsões do ensemble
        """
        self.fit(data)
        return self.predict(steps=horizon)

    def update_weights(self, new_weights: np.ndarray):
        """
        Atualiza pesos do ensemble (usado pelo agente RL).

        Args:
            new_weights: Novos pesos
        """
        assert len(new_weights) == self.n_models, "Número de pesos inválido"
        self.weights = new_weights / new_weights.sum()

    def get_weights(self) -> np.ndarray:
        """
        Retorna os pesos atuais.

        Returns:
            Array com pesos
        """
        return self.weights.copy()

    def evaluate(self, data: Union[np.ndarray, pd.Series], test_size: int = 12) -> Dict:
        """
        Avalia o ensemble em dados de teste.

        Args:
            data: Dados completos
            test_size: Tamanho do conjunto de teste

        Returns:
            Métricas de avaliação
        """
        if isinstance(data, pd.Series):
            data = data.values

        # Divide dados
        train_data = data[:-test_size]
        test_data = data[-test_size:]

        # Treina
        self.fit(train_data)

        # Prevê
        predictions = self.predict(steps=test_size)

        # Calcula métricas
        mse = np.mean((test_data - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_data - predictions))
        mape = np.mean(np.abs((test_data - predictions) / (test_data + 1e-8))) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions,
            'actual': test_data
        }

    def __repr__(self) -> str:
        models_str = ', '.join([m.name for m in self.models])
        return f"EnsemblePredictor(models=[{models_str}], weights={self.weights})"
