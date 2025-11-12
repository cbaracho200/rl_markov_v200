# 📓 Guia de Uso no Google Colab

Este guia explica como usar o Framework de RL para Previsão de Ciclos Econômicos no Google Colab.

## 🚀 Início Rápido (3 opções)

### Opção 1: Usar o Notebook Pronto (Recomendado)

**Link direto para o Colab:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cbaracho200/rl_markov_v200/blob/claude/rl-framework-migration-011CV2sYqGVqNsR2Hue2a5yr/notebooks/colab_example.ipynb)

1. Clique no badge acima
2. Execute as células sequencialmente (Shift+Enter)
3. Aguarde o treinamento (~5-10 minutos)
4. Veja os resultados!

### Opção 2: Código Rápido (1 célula)

Cole este código em uma célula do Colab:

```python
# Setup completo em uma célula
!git clone https://github.com/cbaracho200/rl_markov_v200.git
%cd rl_markov_v200
!git checkout claude/rl-framework-migration-011CV2sYqGVqNsR2Hue2a5yr
!pip install -q torch gymnasium statsmodels pmdarima xgboost pandas numpy matplotlib seaborn tqdm scikit-learn

# Executa exemplo
!python examples/basic_example.py
```

### Opção 3: Instalação Manual

#### 1. Clone e Instale (célula 1)
```python
!git clone https://github.com/cbaracho200/rl_markov_v200.git
%cd rl_markov_v200
!git checkout claude/rl-framework-migration-011CV2sYqGVqNsR2Hue2a5yr
!pip install -q torch gymnasium statsmodels pmdarima xgboost pandas numpy matplotlib seaborn tqdm scikit-learn
```

#### 2. Imports (célula 2)
```python
import sys
sys.path.insert(0, '/content/rl_markov_v200')

from src.utils.data_utils import generate_synthetic_data
from src.models.arima_model import ARIMAPredictor
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.ensemble_predictor import EnsemblePredictor
from src.environments.timeseries_env import TimeSeriesEnv
from src.agents.rl_agent import RLAgent
from src.training.trainer import RLTrainer
```

#### 3. Execute o Pipeline (célula 3)
```python
# Gera dados
data = generate_synthetic_data(n_points=300, seed=42)

# Cria modelos
models = [
    ARIMAPredictor(order=(2, 1, 2)),
    LSTMPredictor(lookback=12, epochs=30),  # Reduzido para Colab
    XGBoostPredictor(lookback=12, n_estimators=50)
]

# Cria ensemble
ensemble = EnsemblePredictor(models)
ensemble.fit(data['value'][:210])  # 70% dos dados

# Cria ambiente e agente
env = TimeSeriesEnv(data.iloc[:210], forecast_horizon=6, n_coefficients=3)
agent = RLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

# Treina
trainer = RLTrainer(env, agent, ensemble)
history = trainer.train(n_episodes=100)  # Reduzido para Colab

# Avalia
results = trainer.evaluate(n_episodes=10)
print(f"\n🎯 MAPE: {results.get('mape', 0):.2f}%")
```

## 💡 Dicas para o Colab

### 1. Ativar GPU (Opcional, mas recomendado para LSTM)

```
Runtime > Change runtime type > Hardware accelerator > GPU
```

Depois, no código:
```python
agent = RLAgent(..., device='cuda')  # Em vez de 'cpu'
```

### 2. Reduzir Tempo de Treinamento

Para testes rápidos:
```python
# Menos episódios no RL
trainer.train(n_episodes=50)

# Menos epochs no LSTM
LSTMPredictor(epochs=20)

# Menos estimadores no XGBoost
XGBoostPredictor(n_estimators=30)
```

### 3. Salvar Modelo no Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Salva agente
agent.save('/content/drive/MyDrive/rl_agent.pt')

# Salva coeficientes
import numpy as np
np.save('/content/drive/MyDrive/best_coefficients.npy', best_coefficients)
```

### 4. Carregar seus Próprios Dados

```python
from google.colab import files
import pandas as pd

# Upload de arquivo
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Carrega dados
data = pd.read_csv(filename)

# Certifique-se de ter uma coluna 'value'
if 'value' not in data.columns:
    data['value'] = data['sua_coluna_aqui']  # Renomeia

# Continue com o pipeline normal...
```

## 🔧 Ajustes Recomendados para Colab

### Configuração Rápida (5-10 minutos)
```python
config = {
    'n_episodes': 50,
    'lstm_epochs': 20,
    'xgb_estimators': 30,
    'max_steps': 30
}
```

### Configuração Balanceada (10-20 minutos)
```python
config = {
    'n_episodes': 100,
    'lstm_epochs': 50,
    'xgb_estimators': 50,
    'max_steps': 50
}
```

### Configuração Completa (20-40 minutos)
```python
config = {
    'n_episodes': 200,
    'lstm_epochs': 100,
    'xgb_estimators': 100,
    'max_steps': 100
}
```

## 📊 Interpretando Resultados

### Métricas

- **MAPE < 5%**: Excelente! 🌟
- **MAPE 5-10%**: Muito bom! ✅
- **MAPE 10-15%**: Bom! 👍
- **MAPE > 15%**: Aceitável (treine por mais tempo) ⚠️

### Exemplo de Saída Esperada

```
================================================================================
RESULTADOS NO CONJUNTO DE TESTE
================================================================================
  MAPE:     4.82%
  RMSE:     3.34
  MAE:      2.89
  R²:       0.91
  Acurácia Direcional: 85.71%
================================================================================

💡 Interpretação: 🌟 EXCELENTE!
```

## ❓ Problemas Comuns

### 1. Erro: "No module named 'src'"
**Solução:**
```python
import sys
sys.path.insert(0, '/content/rl_markov_v200')
```

### 2. Erro: "CUDA out of memory"
**Solução:**
```python
# Use CPU em vez de GPU
agent = RLAgent(..., device='cpu')
```

### 3. Timeout durante treinamento
**Solução:**
```python
# Reduza o número de episódios
trainer.train(n_episodes=50)
```

### 4. Gráficos não aparecem
**Solução:**
```python
import matplotlib.pyplot as plt
plt.show()  # Adicione após cada plot
```

## 📚 Recursos Adicionais

- **Documentação Completa**: [README.md](README.md)
- **Exemplos**: [examples/](examples/)
- **Código Fonte**: [src/](src/)

## 🆘 Suporte

Problemas ou dúvidas? Abra uma [issue no GitHub](https://github.com/cbaracho200/rl_markov_v200/issues)!

---

**Happy Forecasting! 🚀📈**
