# 🐛 Troubleshooting - Erros Comuns e Soluções

Este guia resolve os erros mais comuns ao usar o framework.

---

## ❌ Erro 1: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

### Mensagem Completa:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x37 and 44x256)
```

### Causa:
Você está criando o agente com dimensões **hardcoded** que não correspondem ao ambiente.

### ❌ ERRADO:
```python
agent = RLAgent(
    state_dim=44,      # ❌ Valor hardcoded!
    action_dim=10      # ❌ Valor hardcoded!
)
```

### ✅ CORRETO:
```python
# SEMPRE pegue as dimensões do ambiente
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = RLAgent(
    state_dim=state_dim,    # ✅ Correto!
    action_dim=action_dim   # ✅ Correto!
)
```

### Explicação:
O ambiente calcula o estado baseado em:
- `window_size` (24 por padrão)
- `n_coefficients` (número de modelos)
- `statistical_features` (10 features)

**Total**: 24 + n_modelos + 10

Se você tem 3 modelos: 24 + 3 + 10 = **37** (não 44!)

---

## ❌ Erro 2: `ImportError: cannot import name 'AdvancedRLAgent'`

### Mensagem Completa:
```
ImportError: cannot import name 'AdvancedRLAgent' from 'src.agents'
```

### Causas Possíveis:

#### 1. Código Desatualizado no Colab
O Colab pode ter uma versão antiga em cache.

**Solução:**
```python
# Opção A: Reinicie o runtime
# Runtime > Restart Runtime

# Opção B: Force reload dos módulos
import importlib
import src.agents
importlib.reload(src.agents)

# Opção C: Reimporte o repositório
!rm -rf Previs-o-ciclos-Econ-mico
!git clone https://github.com/cbaracho200/Previs-o-ciclos-Econ-mico.git
%cd Previs-o-ciclos-Econ-mico
```

#### 2. Repositório Não Atualizado
Você pode ter clonado antes do agente avançado ser adicionado.

**Solução:**
```bash
# Atualize o repositório
!cd Previs-o-ciclos-Econ-mico && git pull origin main
```

#### 3. Import Incorreto
**Solução:**
```python
# ✅ Forma correta de importar
from src.agents import AdvancedRLAgent
from src.training import AdvancedRLTrainer

# Ou importe diretamente
from src.agents.rl_agent_advanced import AdvancedRLAgent
```

---

## ❌ Erro 3: `ARIMA.fit() got an unexpected keyword argument 'disp'`

### Mensagem Completa:
```
Erro ao treinar ARIMA: ARIMA.fit() got an unexpected keyword argument 'disp'
✗ ARIMA falhou no treinamento
```

### Causa:
Versão antiga do código com parâmetro `disp=False` obsoleto.

### Solução:
```bash
# Atualize o repositório
!cd Previs-o-ciclos-Econ-mico && git pull origin main

# Ou verifique se arima_model.py tem:
# ✅ self.fitted_model = self.model.fit()
# ❌ self.fitted_model = self.model.fit(disp=False)  # REMOVA disp=False
```

---

## ❌ Erro 4: `ValueError: setting an array element with a sequence`

### Mensagem Completa:
```
ValueError: setting an array element with a sequence.
The requested array has an inhomogeneous shape after 1 dimensions.
```

### Causa:
Bug no método `compute_gae()` do agente RL (já corrigido).

### Solução:
```bash
# Atualize o repositório
!cd Previs-o-ciclos-Econ-mico && git pull origin main
```

Ou verifique manualmente em `src/agents/rl_agent.py`:
- Linha 228: deve usar `.item()` não `.squeeze()`
- Linha 295: deve usar `.item()` não `.squeeze()`

---

## ❌ Erro 5: `KeyError: 'value'` ao usar seus dados

### Causa:
O DataFrame precisa ter uma coluna chamada `'value'`.

### Solução:
```python
# Renomeie sua coluna de valores
data = data.rename(columns={'sua_coluna': 'value'})

# Ou especifique a coluna ao criar o ambiente
env = TimeSeriesEnv(
    data=data[['data', 'sua_coluna']].rename(columns={'sua_coluna': 'value'}),
    forecast_horizon=12
)
```

---

## ❌ Erro 6: `RuntimeError: CUDA out of memory`

### Causa:
Tentando usar GPU mas modelo é muito grande para a memória disponível.

### Solução:
```python
# Opção 1: Use CPU
agent = AdvancedRLAgent(
    ...,
    device='cpu'
)

# Opção 2: Reduza o tamanho do modelo
agent = AdvancedRLAgent(
    ...,
    hidden_dim=128,    # Reduzido de 512
    num_heads=4,       # Reduzido de 8
    num_layers=2,      # Reduzido de 3
    device='cuda'
)

# Opção 3: Reduza batch size
history = trainer.train(
    ...,
    batch_size=32      # Reduzido de 64
)
```

---

## ❌ Erro 7: Treinamento muito lento no Colab

### Soluções:

#### 1. Reduza número de episódios
```python
history = trainer.train(
    n_episodes=100,    # Em vez de 500
    max_steps=50
)
```

#### 2. Use agente padrão em vez do avançado
```python
# Use RLAgent (mais rápido) em vez de AdvancedRLAgent
agent = RLAgent(...)
trainer = RLTrainer(...)
```

#### 3. Ative GPU no Colab
```
Runtime > Change runtime type > Hardware accelerator > GPU
```

Então use:
```python
agent = RLAgent(..., device='cuda')
```

---

## ❌ Erro 8: `ModuleNotFoundError: No module named 'gymnasium'`

### Solução:
```bash
# Instale as dependências
!pip install gymnasium torch statsmodels xgboost pandas numpy matplotlib seaborn tqdm scikit-learn
```

---

## ❌ Erro 9: MAPE muito alto (> 20%)

### Possíveis Causas:

#### 1. Poucos episódios de treinamento
```python
# Aumente n_episodes
history = trainer.train(n_episodes=200)  # Em vez de 100
```

#### 2. Learning rate inadequado
```python
agent = RLAgent(
    ...,
    learning_rate=1e-4  # Tente 1e-3 ou 3e-4
)
```

#### 3. Dados insuficientes
```python
# Use pelo menos 200 pontos de treino
data = generate_synthetic_data(n_points=300)  # Aumente se necessário
```

#### 4. Modelos base não treinados corretamente
```python
# Verifique se ensemble.fit() foi chamado
ensemble.fit(train_data['value'])
```

---

## ❌ Erro 10: `AttributeError: 'NoneType' object has no attribute 'shape'`

### Causa:
Tentando usar coeficientes que não foram encontrados.

### Solução:
```python
best_coefficients = trainer.get_best_coefficients()

if best_coefficients is not None:
    ensemble.update_weights(best_coefficients)
else:
    print("⚠️ Usando pesos iguais")
    # Continue com pesos iguais
```

---

## 📞 Ainda com Problemas?

### Checklist de Debug:

1. ✅ Repositório atualizado?
   ```bash
   !cd Previs-o-ciclos-Econ-mico && git pull
   ```

2. ✅ Dependências instaladas?
   ```bash
   !pip install -q torch gymnasium statsmodels xgboost
   ```

3. ✅ Runtime reiniciado?
   ```
   Runtime > Restart Runtime
   ```

4. ✅ Usando dimensões corretas?
   ```python
   state_dim = env.observation_space.shape[0]
   action_dim = env.action_space.shape[0]
   ```

5. ✅ Dados têm coluna 'value'?
   ```python
   print(data.columns)
   ```

### Abra uma Issue:

Se nenhuma solução funcionou, abra uma issue no GitHub com:
- ✅ Mensagem de erro completa
- ✅ Código que está executando
- ✅ Versão do Python (`!python --version`)
- ✅ Versões das bibliotecas (`!pip list`)

**Link**: https://github.com/cbaracho200/Previs-o-ciclos-Econ-mico/issues

---

## 💡 Dicas Gerais

### 1. Sempre use print para debug
```python
print(f"State dim: {env.observation_space.shape[0]}")
print(f"Action dim: {env.action_space.shape[0]}")
print(f"Data shape: {data.shape}")
print(f"Has 'value' column: {'value' in data.columns}")
```

### 2. Comece simples
```python
# Teste primeiro com configuração mínima
agent = RLAgent(state_dim=state_dim, action_dim=action_dim)
history = trainer.train(n_episodes=10)  # Só 10 para testar
```

### 3. Salve checkpoints
```python
# Salve periodicamente para não perder progresso
trainer.train(
    n_episodes=200,
    save_frequency=50  # Salva a cada 50 episódios
)
```

---

**Última atualização**: 2025-01-12
**Versão do Framework**: 2.0.0
