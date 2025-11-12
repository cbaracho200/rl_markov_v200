# 🎓 Modelos Avançados & Otimização Recursiva

## Visão Geral

Este framework agora inclui **4 modelos supervisionados state-of-the-art** e **otimização automática de hiperparâmetros com Optuna**.

---

## 🤖 Novos Modelos Supervisionados

### 1. **AutoARIMA** (`AutoARIMAPredictor`)

**Descrição**: ARIMA com busca automática de hiperparâmetros usando pmdarima.

**Vantagens**:
- ✅ Encontra automaticamente os melhores parâmetros (p, d, q)
- ✅ Suporta sazonalidade automática (SARIMA)
- ✅ Usa testes estatísticos (ADF, KPSS)
- ✅ Mais robusto que ARIMA manual

**Uso**:
```python
from src.models import AutoARIMAPredictor

model = AutoARIMAPredictor(
    max_p=5,              # Máximo AR order
    max_d=2,              # Máximo differencing
    max_q=5,              # Máximo MA order
    seasonal=True,        # Usa SARIMA
    m=12,                 # Período sazonal (12 para mensal)
    stepwise=True,        # Busca stepwise (mais rápido)
    information_criterion='aic',  # 'aic', 'bic', ou 'hqic'
    trace=False           # Mostra progresso
)

model.fit(train_data)
predictions = model.predict(steps=12)

# Acessa melhores parâmetros
best_params = model.get_best_parameters()
print(f"Melhor ordem: {best_params['order']}")
print(f"AIC: {best_params['aic']}")
```

**Quando usar**:
- Séries com padrões lineares e sazonalidade clara
- Quando não sabe os melhores parâmetros ARIMA
- Dados com tendência e sazonalidade

**Performance**: MAPE típico: 5-15%

---

### 2. **Prophet** (`ProphetPredictor`)

**Descrição**: Modelo do Facebook robusto a outliers e dados faltantes.

**Vantagens**:
- ✅ Robusto a dados faltantes e outliers
- ✅ Múltiplas sazonalidades (diária, semanal, anual)
- ✅ Detecta mudanças de tendência automaticamente
- ✅ Suporta feriados e eventos especiais
- ✅ Interpretável (componentes separados)

**Uso**:
```python
from src.models import ProphetPredictor

model = ProphetPredictor(
    seasonality_mode='multiplicative',  # ou 'additive'
    yearly_seasonality='auto',          # ou True/False
    weekly_seasonality='auto',
    daily_seasonality='auto',
    changepoint_prior_scale=0.05,       # Flexibilidade da tendência (0.001-0.5)
    seasonality_prior_scale=10.0        # Força da sazonalidade (0.01-10)
)

model.fit(train_data)
predictions = model.predict(steps=12)
```

**Quando usar**:
- Séries com forte sazonalidade
- Dados com outliers ou valores faltantes
- Múltiplas sazonalidades (ex: vendas com padrões semanais e anuais)
- Quando precisa de interpretabilidade

**Performance**: MAPE típico: 5-12%

**Hiperparâmetros chave**:
- `changepoint_prior_scale`: ↑ = mais flexível, ↓ = mais suave
- `seasonality_mode`: 'multiplicative' para séries com sazonalidade crescente

---

### 3. **CatBoost** (`CatBoostPredictor`)

**Descrição**: Gradient boosting state-of-the-art da Yandex.

**Vantagens**:
- ✅ Melhor performance que XGBoost em muitos casos
- ✅ Suporte nativo a features categóricas
- ✅ Menor overfitting
- ✅ Treinamento mais rápido
- ✅ Não requer normalização de dados

**Uso**:
```python
from src.models import CatBoostPredictor

model = CatBoostPredictor(
    lookback=12,                  # Número de lags
    iterations=500,               # Número de árvores
    learning_rate=0.03,           # Taxa de aprendizado
    depth=6,                      # Profundidade das árvores
    l2_leaf_reg=3.0,              # Regularização L2
    random_strength=1.0,          # Força da aleatoriedade
    bagging_temperature=1.0       # Temperatura do bagging
)

model.fit(train_data)
predictions = model.predict(steps=12)
```

**Quando usar**:
- Dados com relações não-lineares complexas
- Quando XGBoost ou LightGBM estão sofrendo overfitting
- Dados com features categóricas
- Produção (treinamento rápido)

**Performance**: MAPE típico: 3-10%

**Hiperparâmetros chave**:
- `iterations`: 300-1000 (mais = melhor, mas mais lento)
- `learning_rate`: 0.01-0.1 (↓ para evitar overfitting)
- `depth`: 4-10 (↑ = mais complexo, risco de overfit)

---

### 4. **LightGBM** (`LightGBMPredictor`)

**Descrição**: Gradient boosting ultra-rápido da Microsoft.

**Vantagens**:
- ✅ **Extremamente rápido** (10-100x mais que XGBoost)
- ✅ Baixíssimo uso de memória
- ✅ Excelente para datasets grandes (>10k pontos)
- ✅ Suporta missing values nativamente
- ✅ Paralelização eficiente

**Uso**:
```python
from src.models import LightGBMPredictor

model = LightGBMPredictor(
    lookback=12,                  # Número de lags
    n_estimators=500,             # Número de árvores
    learning_rate=0.05,           # Taxa de aprendizado
    num_leaves=31,                # Número de folhas (específico do LightGBM)
    max_depth=-1,                 # -1 = sem limite
    min_child_samples=20,         # Mínimo de samples nas folhas
    subsample=0.8,                # Fração de samples
    colsample_bytree=0.8,         # Fração de features
    reg_alpha=0.1,                # Regularização L1
    reg_lambda=0.1                # Regularização L2
)

model.fit(train_data)
predictions = model.predict(steps=12)
```

**Quando usar**:
- Datasets grandes (>10,000 pontos)
- Quando velocidade é crítica
- Recursos computacionais limitados
- Produção em larga escala

**Performance**: MAPE típico: 3-10%

**Hiperparâmetros chave**:
- `num_leaves`: 20-50 (específico do LightGBM, controla complexidade)
- `n_estimators`: 300-1000
- `learning_rate`: 0.01-0.1

---

## 🔍 Otimização Automática de Hiperparâmetros

### **HyperparameterOptimizer**

Usa **Optuna** (Bayesian Optimization) para encontrar os melhores hiperparâmetros.

**Vantagens sobre Grid Search**:
- ✅ **Muito mais eficiente** (10-100x menos trials)
- ✅ Aprende com trials anteriores (Bayesian)
- ✅ Pruning automático de trials ruins
- ✅ Suporta otimização paralela

**Uso Básico**:
```python
from src.optimization import HyperparameterOptimizer
from src.models import CatBoostPredictor

# Cria otimizador
optimizer = HyperparameterOptimizer(
    metric='mape',           # Métrica a minimizar
    direction='minimize',    # ou 'maximize'
    n_trials=50,             # Número de trials
    verbose=True
)

# Define espaço de busca
param_space = {
    'lookback': ('int', 6, 24),                    # (tipo, min, max)
    'iterations': ('int', 100, 500),
    'learning_rate': ('float', 0.01, 0.1, 'log'), # 'log' = escala logarítmica
    'depth': ('int', 4, 10)
}

# Otimiza
best_params = optimizer.optimize_model(
    model_class=CatBoostPredictor,
    train_data=train_data,
    val_data=val_data,
    param_space=param_space,
    forecast_horizon=12
)

# Usa melhores parâmetros
model = CatBoostPredictor(**best_params)
```

**Otimização de Múltiplos Modelos**:
```python
# Define configurações de múltiplos modelos
model_configs = [
    {
        'class': CatBoostPredictor,
        'param_space': {
            'lookback': ('int', 6, 24),
            'iterations': ('int', 100, 500),
            'learning_rate': ('float', 0.01, 0.1, 'log')
        }
    },
    {
        'class': LightGBMPredictor,
        'param_space': {
            'lookback': ('int', 6, 24),
            'n_estimators': ('int', 100, 500),
            'learning_rate': ('float', 0.01, 0.1, 'log')
        }
    }
]

# Otimiza todos
all_best_params = optimizer.optimize_ensemble(
    model_configs=model_configs,
    train_data=train_data,
    val_data=val_data,
    forecast_horizon=12
)
```

---

### **RecursiveOptimizer**

Reotimiza hiperparâmetros **durante o treinamento** se a performance estagnar.

**Como Funciona**:
1. Monitora performance a cada N episódios
2. Se não melhorar o suficiente, reotimiza hiperparâmetros
3. Atualiza modelos com novos parâmetros
4. Continua treinamento

**Uso**:
```python
from src.optimization import HyperparameterOptimizer, RecursiveOptimizer

# Cria otimizadores
hp_optimizer = HyperparameterOptimizer(n_trials=20)

recursive_opt = RecursiveOptimizer(
    hyperparameter_optimizer=hp_optimizer,
    reoptimize_frequency=50,      # Reotimiza a cada 50 episódios
    performance_window=20,         # Janela para calcular performance
    improvement_threshold=0.05     # Reotimiza se melhoria < 5%
)

# Durante o treinamento
for episode in range(n_episodes):
    # ... treina episódio ...

    # Checa se deve reotimizar
    if recursive_opt.should_reoptimize(current_performance):
        print("🔄 Reotimizando hiperparâmetros...")

        new_params = recursive_opt.reoptimize(
            model_configs=model_configs,
            train_data=recent_train_data,
            val_data=val_data
        )

        # Atualiza modelos
        for model_name, params in new_params.items():
            # Recria modelo com novos parâmetros
            ...
```

---

## 📊 Comparação de Modelos

| Modelo | Velocidade | Precisão | Interpretabilidade | Quando Usar |
|--------|------------|----------|-------------------|-------------|
| **ARIMA** | ⚡⚡⚡ | ★★★ | ★★★★★ | Séries lineares, sazonalidade simples |
| **AutoARIMA** | ⚡⚡ | ★★★★ | ★★★★★ | ARIMA sem saber parâmetros |
| **Prophet** | ⚡⚡⚡ | ★★★★ | ★★★★★ | Outliers, dados faltantes, múltiplas sazonalidades |
| **LSTM** | ⚡ | ★★★★ | ★★ | Dependências longas, padrões complexos |
| **XGBoost** | ⚡⚡ | ★★★★ | ★★★ | Relações não-lineares |
| **CatBoost** | ⚡⚡ | ★★★★★ | ★★★ | Features categóricas, menos overfitting |
| **LightGBM** | ⚡⚡⚡⚡⚡ | ★★★★ | ★★★ | Datasets grandes, produção |

---

## 💡 Recomendações de Uso

### **Para Máxima Precisão**:
```python
models = [
    AutoARIMAPredictor(seasonal=True),
    ProphetPredictor(seasonality_mode='multiplicative'),
    CatBoostPredictor(iterations=500, depth=8),
    LightGBMPredictor(n_estimators=500, num_leaves=40)
]

# Otimiza hiperparâmetros
optimizer = HyperparameterOptimizer(n_trials=50)
# ... otimiza ...

# Usa ensemble com RL
ensemble = EnsemblePredictor(models)
# ... treina com RL ...
```

### **Para Velocidade (Produção)**:
```python
models = [
    LightGBMPredictor(n_estimators=200),  # Muito rápido
    ProphetPredictor(),                   # Rápido e robusto
    AutoARIMAPredictor(stepwise=True)     # Busca rápida
]
```

### **Para Interpretabilidade**:
```python
models = [
    AutoARIMAPredictor(trace=True),       # Mostra parâmetros
    ProphetPredictor()                    # Componentes separados
]
```

---

## 🚀 Exemplo Completo

```python
from src.models import AutoARIMAPredictor, ProphetPredictor, CatBoostPredictor, LightGBMPredictor
from src.models import EnsemblePredictor
from src.optimization import HyperparameterOptimizer
from src.agents import AdvancedRLAgent
from src.training import AdvancedRLTrainer

# 1. Cria modelos avançados
models = [
    AutoARIMAPredictor(name="AutoARIMA"),
    ProphetPredictor(name="Prophet"),
    CatBoostPredictor(iterations=300, name="CatBoost"),
    LightGBMPredictor(n_estimators=300, name="LightGBM")
]

# 2. Otimiza hiperparâmetros
optimizer = HyperparameterOptimizer(metric='mape', n_trials=30)

model_configs = [
    {
        'class': CatBoostPredictor,
        'param_space': {
            'iterations': ('int', 100, 500),
            'learning_rate': ('float', 0.01, 0.1, 'log'),
            'depth': ('int', 4, 10)
        }
    },
    {
        'class': LightGBMPredictor,
        'param_space': {
            'n_estimators': ('int', 100, 500),
            'learning_rate': ('float', 0.01, 0.1, 'log'),
            'num_leaves': ('int', 20, 50)
        }
    }
]

best_params = optimizer.optimize_ensemble(
    model_configs, train_data, val_data, forecast_horizon=12
)

# 3. Cria modelos otimizados
optimized_models = [
    AutoARIMAPredictor(),
    ProphetPredictor(),
    CatBoostPredictor(**best_params['CatBoostPredictor']),
    LightGBMPredictor(**best_params['LightGBMPredictor'])
]

# 4. Cria ensemble
ensemble = EnsemblePredictor(optimized_models)
ensemble.fit(train_data)

# 5. Treina agente RL avançado
agent = AdvancedRLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=len(optimized_models),
    use_per=True,
    use_transformer=True
)

trainer = AdvancedRLTrainer(env, agent, ensemble)
history = trainer.train(n_episodes=200)

# 6. Avalia
results = trainer.evaluate(n_episodes=10)
print(f"MAPE: {results['mape']:.2f}%")
```

---

## 📦 Dependências

Instale as novas dependências:

```bash
pip install prophet catboost lightgbm pmdarima optuna plotly
```

Ou instale todas de uma vez:

```bash
pip install -r requirements.txt
```

---

## 🎯 Próximos Passos

1. **Execute o exemplo**: `python examples/advanced_models_example.py`
2. **Otimize seus modelos**: Use `HyperparameterOptimizer`
3. **Experimente diferentes combinações**: Teste diferentes ensembles
4. **Use otimização recursiva**: Para máxima performance

---

## 📚 Referências

- **AutoARIMA**: [pmdarima docs](https://alkaline-ml.com/pmdarima/)
- **Prophet**: [Facebook Prophet](https://facebook.github.io/prophet/)
- **CatBoost**: [CatBoost docs](https://catboost.ai/)
- **LightGBM**: [LightGBM docs](https://lightgbm.readthedocs.io/)
- **Optuna**: [Optuna docs](https://optuna.readthedocs.io/)

---

**Desenvolvido com ❤️ para previsão de ciclos econômicos com técnicas state-of-the-art**
