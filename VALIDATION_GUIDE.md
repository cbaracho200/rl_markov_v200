# Guia de Validação de Variáveis Preditoras

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Level: PhD](https://img.shields.io/badge/level-PhD-red.svg)](.)

**Sistema completo de validação estatística de variáveis preditoras para séries temporais, incluindo testes de estacionaridade e causalidade de Granger.**

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Testes Implementados](#-testes-implementados)
  - [1. Testes de Estacionaridade](#1-testes-de-estacionaridade)
  - [2. Testes de Causalidade de Granger](#2-testes-de-causalidade-de-granger)
- [Pipeline de Validação](#-pipeline-de-validação)
- [Modelos Avançados](#-modelos-avançados)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Interpretação dos Resultados](#-interpretação-dos-resultados)

---

## 🎯 Visão Geral

Antes de treinar modelos de previsão, é **essencial validar estatisticamente** as variáveis preditoras. Este módulo implementa um pipeline completo de validação que:

1. **Verifica estacionaridade** de todas as variáveis
2. **Transforma** séries não-estacionárias (diferenciação)
3. **Testa causalidade de Granger** para identificar preditores válidos
4. **Seleciona automaticamente** variáveis com relações causais significativas

### Por que isso é importante?

❌ **Sem validação:**
- Modelos podem usar preditores irrelevantes
- Séries não-estacionárias causam previsões espúrias
- Relações podem ser coincidências estatísticas

✅ **Com validação:**
- Apenas preditores com causalidade de Granger são usados
- Séries estacionárias garantem previsões confiáveis
- Relações são estatisticamente significativas

---

## 🔬 Testes Implementados

### 1. Testes de Estacionaridade

#### 1.1 Teste ADF (Augmented Dickey-Fuller)

**Hipóteses:**
- H₀: Série tem raiz unitária (não estacionária)
- H₁: Série é estacionária

**Interpretação:**
- p-value < 0.05 → **Rejeitar H₀** → Série é estacionária ✓

**Quando usar:**
- Teste mais comum para estacionaridade
- Robusto para séries com tendência

```python
from src.validation import StationarityTests

tester = StationarityTests()
result = tester.adf_test(data)

print(f"P-value: {result['p_value']:.6f}")
print(f"Conclusão: {result['conclusion']}")
```

#### 1.2 Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

**Hipóteses:**
- H₀: Série é estacionária
- H₁: Série não é estacionária

⚠️ **ATENÇÃO:** Hipótese nula OPOSTA ao ADF!

**Interpretação:**
- p-value > 0.05 → **Não rejeitar H₀** → Série é estacionária ✓

**Quando usar:**
- Complementa o teste ADF
- Mais sensível a estacionaridade em tendência

```python
result = tester.kpss_test(data)

print(f"P-value: {result['p_value']:.6f}")
print(f"Conclusão: {result['conclusion']}")
```

#### 1.3 Teste Phillips-Perron

**Hipóteses:**
- H₀: Série tem raiz unitária (não estacionária)
- H₁: Série é estacionária

**Interpretação:**
- p-value < 0.05 → **Rejeitar H₀** → Série é estacionária ✓

**Quando usar:**
- Similar ao ADF, mas mais robusto a heterocedasticidade
- Recomendado para dados com volatilidade variável

```python
result = tester.phillips_perron_test(data)

print(f"P-value: {result['p_value']:.6f}")
print(f"Conclusão: {result['conclusion']}")
```

#### 1.4 Executar Todos os Testes (Recomendado)

```python
# Executa ADF, KPSS e Phillips-Perron
results = tester.run_all_tests(data, verbose=True)

# Decisão por consenso (2/3)
print(f"Consenso: {results['consensus']['is_stationary']}")
print(f"Votos: {results['consensus']['votes']}")
```

**Exemplo de saída:**

```
================================================================================
                    RESULTADOS DOS TESTES DE ESTACIONARIDADE
================================================================================

1. TESTE ADF (Augmented Dickey-Fuller)
--------------------------------------------------------------------------------
Estatística de teste: -4.582316
P-valor: 0.000123
Lags utilizados: 12
Observações: 287

Conclusão: Estacionária

✓ Forte evidência de estacionaridade (p < 0.01)
  Estatística (-4.5823) < Valor Crítico 1% (-3.4500)

2. TESTE KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
--------------------------------------------------------------------------------
Estatística de teste: 0.234567
P-valor: 0.100000
Lags utilizados: 8

Conclusão: Estacionária

✓ Forte evidência de estacionaridade (p > 0.10)

3. TESTE PHILLIPS-PERRON
--------------------------------------------------------------------------------
Estatística de teste: -4.891234
P-valor: 0.000045

Conclusão: Estacionária

✓ Forte evidência de estacionaridade (p < 0.01)

================================================================================
                              CONSENSO DOS TESTES
================================================================================
Votos pela estacionaridade: 3/3
Decisão: ESTACIONÁRIA

✓ SÉRIE ESTACIONÁRIA: Todos os testes confirmam. Pode usar diretamente em modelos.
================================================================================
```

#### 1.5 Transformação para Estacionaridade

Se a série não é estacionária, pode ser transformada via **diferenciação**:

```python
from src.validation import make_stationary

stationary_series, diff_order = make_stationary(data, max_diff=2)

print(f"Ordem de diferenciação: {diff_order}")
# Ordem 0: Já é estacionária
# Ordem 1: Primeira diferença (yt - yt-1)
# Ordem 2: Segunda diferença
```

---

### 2. Testes de Causalidade de Granger

#### O que é Causalidade de Granger?

**X "Granger-causa" Y** se:
- Valores passados de X contêm informação útil para prever Y
- Além da informação contida nos valores passados de Y

⚠️ **IMPORTANTE:** Não implica causalidade real, apenas **precedência temporal útil para previsão**.

#### 2.1 Teste Bivariado (X → Y)

**Hipóteses:**
- H₀: X **NÃO** Granger-causa Y
- H₁: X Granger-causa Y

**Interpretação:**
- p-value < 0.05 → **Rejeitar H₀** → X Granger-causa Y ✓

```python
from src.validation import GrangerCausality

gc = GrangerCausality(max_lag=12)

# Testar se Interest_Rate causa GDP
result = gc.test_granger_causality(
    data,
    x_var='Interest_Rate',
    y_var='GDP',
    verbose=True
)

print(f"Granger-causa? {result['granger_causes']}")
print(f"Melhor lag: {result['best_lag']}")
print(f"P-value: {result['best_p_value']:.6f}")
print(f"Força: {result['strength']}")
```

**Exemplo de saída:**

```
--------------------------------------------------------------------------------
Teste de Causalidade de Granger: Interest_Rate → GDP
--------------------------------------------------------------------------------
Conclusão: ✓ Interest_Rate Granger-causa GDP (lag=1, p=0.000234)
Melhor lag: 1
P-value: 0.000234
Estatística F: 18.4567
Força da relação: very_strong

P-values por lag:
  Lag 1: 0.000234 ***
  Lag 2: 0.001234 **
  Lag 3: 0.045678 *
  ...
```

#### 2.2 Testar Todas as Combinações

```python
# Testar todas as variáveis → GDP
results = gc.test_all_combinations(
    data,
    target_var='GDP',
    verbose=True
)

print(f"Relações significativas: {len(results['significant_relationships'])}")
```

#### 2.3 Seleção Automática de Preditores

```python
# Seleciona apenas preditores com causalidade significativa
selected = gc.select_predictors(
    data,
    target_var='GDP',
    min_strength='weak',  # 'weak', 'moderate', 'strong', 'very_strong'
    verbose=True
)

print(f"Preditores selecionados: {selected}")
```

**Exemplo de saída:**

```
================================================================================
                   SELEÇÃO DE PREDITORES PARA: GDP
================================================================================
Força mínima: weak

Variáveis selecionadas: 3
--------------------------------------------------------------------------------
1. Interest_Rate
   P-value: 0.000234
   Lag ótimo: 1
   Força: very_strong

2. Unemployment
   P-value: 0.002456
   Lag ótimo: 1
   Força: strong

3. Consumer_Confidence
   P-value: 0.048912
   Lag ótimo: 2
   Força: weak
================================================================================
```

---

## 🔄 Pipeline de Validação

### Uso Recomendado: `VariableValidator`

O `VariableValidator` integra **todos os testes** em um pipeline automatizado:

```python
from src.validation import VariableValidator

# Inicializar validador
validator = VariableValidator(
    significance_level=0.05,      # 5%
    max_lag_granger=12,            # Testar até 12 lags
    min_causal_strength='weak',    # Aceitar causalidade fraca ou superior
    auto_transform=True            # Transformar automaticamente não-estacionárias
)

# Executar validação completa
results = validator.validate_all(
    data=data,
    target_var='GDP',
    verbose=True
)
```

### Etapas do Pipeline

#### Etapa 1: Testes de Estacionaridade
```
Testando: GDP
  ✗ NÃO ESTACIONÁRIA (consenso: 2/3)

Testando: Interest_Rate
  ✓ ESTACIONÁRIA (consenso: 3/3)

Testando: Inflation
  ✗ NÃO ESTACIONÁRIA (consenso: 3/3)
```

#### Etapa 2: Transformação para Estacionaridade
```
Transformando: GDP
  ✓ Aplicada diferenciação de ordem 1

Transformando: Inflation
  ✓ Aplicada diferenciação de ordem 1

Dados após transformação: 299 observações
```

#### Etapa 3: Testes de Causalidade de Granger
```
Testando: Interest_Rate → GDP
  ✓ Interest_Rate Granger-causa GDP
    Lag ótimo: 1, p-value: 0.000234

Testando: Inflation → GDP
  ✓ Inflation Granger-causa GDP
    Lag ótimo: 2, p-value: 0.003456

Testando: Random_Noise → GDP
  ✗ Random_Noise NÃO Granger-causa GDP
    p-value: 0.734521
```

#### Etapa 4: Seleção de Preditores Válidos
```
✓ 3 preditores válidos selecionados:

1. Interest_Rate
   Causalidade de Granger: p = 0.000234 (very_strong)
   Lag ótimo: 1
   Estatística F: 18.4567
   Estacionária: Sim

2. Unemployment
   Causalidade de Granger: p = 0.002456 (strong)
   Lag ótimo: 1
   Estatística F: 12.3456
   Estacionária: Sim

3. Consumer_Confidence
   Causalidade de Granger: p = 0.048912 (weak)
   Lag ótimo: 2
   Estatística F: 3.9876
   Estacionária: Transformada
```

### Obter Resultados

```python
# Dados transformados + preditores validados
transformed_data, selected_predictors = validator.get_validated_data()

# Importância das features
importance = validator.get_feature_importance()
print(importance)
```

**Saída:**

```
   Rank           Variable  Importance_Score  P_Value      Strength  Lag  F_Statistic
      1      Interest_Rate          0.999766  0.000234  very_strong    1      18.4567
      2       Unemployment          0.997544  0.002456       strong    1      12.3456
      3  Consumer_Confidence        0.951088  0.048912         weak    2       3.9876
```

---

## 🚀 Modelos Avançados

Após validar os preditores, use modelos avançados que incorporam variáveis exógenas:

### 1. SARIMA (Seasonal ARIMA)

**Características:**
- Captura sazonalidade
- Modelo univariado (sem preditores)
- Ideal como baseline

**Notação:** SARIMA(p,d,q)(P,D,Q)s

```python
from src.models import SARIMAPredictor

sarima = SARIMAPredictor(
    order=(1, 1, 1),              # (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s)
)

sarima.fit(train_data['GDP'])
forecast = sarima.predict(steps=12)

# Com intervalos de confiança
intervals = sarima.predict_with_intervals(steps=12, alpha=0.05)
```

### 2. SARIMAX (SARIMA with eXogenous variables)

**Características:**
- SARIMA + variáveis exógenas
- **USE OS PREDITORES VALIDADOS!**
- Melhor desempenho que SARIMA univariado

**Notação:** SARIMAX(p,d,q)(P,D,Q)s + X

```python
from src.models import SARIMAXPredictor

# Usar preditores validados
selected = ['Interest_Rate', 'Unemployment', 'Consumer_Confidence']

sarimax = SARIMAXPredictor(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    exog_names=selected
)

# Treinar com variáveis exógenas
sarimax.fit(
    train_data['GDP'],
    exog=train_data[selected]
)

# Prever (IMPORTANTE: fornecer valores futuros de exog!)
forecast = sarimax.predict(
    steps=12,
    exog=test_data[selected]
)

# Coeficientes das exógenas
coeffs = sarimax.get_exog_coefficients()
print(coeffs)
# {'Interest_Rate': -1.2345, 'Unemployment': -0.6789, ...}
```

### 3. VAR (Vector Autoregression)

**Características:**
- Modelo multivariado
- Modela múltiplas séries simultaneamente
- Captura interdependências e feedback loops
- **Todas as variáveis DEVEM ser estacionárias!**

**Notação:** VAR(p)

```python
from src.models import VARPredictor

# Usar variável alvo + top preditores
var_variables = ['GDP', 'Interest_Rate', 'Unemployment']

var = VARPredictor(maxlags=None, ic='aic')  # Seleção automática de lag

# Treinar com múltiplas variáveis
var.fit(train_data[var_variables])

# Prever todas as variáveis simultaneamente
forecast_all = var.predict(steps=12)

# Ou apenas GDP
forecast_gdp = var.predict_single_variable('GDP', steps=12)

# Análises avançadas
causality_matrix = var.get_granger_causality_matrix()
fevd = var.get_forecast_error_variance_decomposition(periods=10)
```

---

## 💡 Exemplos de Uso

### Exemplo 1: Pipeline Completo

```python
from src.validation import VariableValidator
from src.models import SARIMAXPredictor
import pandas as pd

# 1. Carregar dados
data = pd.read_csv('economic_data.csv')

# 2. Validar variáveis
validator = VariableValidator(
    significance_level=0.05,
    max_lag_granger=12,
    min_causal_strength='weak',
    auto_transform=True
)

results = validator.validate_all(
    data=data,
    target_var='GDP',
    verbose=True
)

# 3. Obter preditores validados
selected_predictors = [p['variable'] for p in results['selected_predictors']]
print(f"Preditores selecionados: {selected_predictors}")

# 4. Treinar modelo SARIMAX com preditores validados
train_size = int(0.8 * len(data))
train = data.iloc[:train_size]
test = data.iloc[train_size:]

sarimax = SARIMAXPredictor(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    exog_names=selected_predictors
)

sarimax.fit(train['GDP'], exog=train[selected_predictors])

# 5. Prever
forecast = sarimax.predict(steps=len(test), exog=test[selected_predictors])

# 6. Avaliar
import numpy as np
actual = test['GDP'].values
mape = np.mean(np.abs((actual - forecast) / actual)) * 100
print(f"MAPE: {mape:.2f}%")
```

### Exemplo 2: Comparar Modelos

```python
from src.models import SARIMAPredictor, SARIMAXPredictor, VARPredictor

# Validar preditores
validator = VariableValidator()
results = validator.validate_all(data, target_var='GDP')
selected = [p['variable'] for p in results['selected_predictors']]

# Treinar múltiplos modelos
models = {
    'SARIMA': SARIMAPredictor(order=(1,1,1), seasonal_order=(1,1,1,12)),
    'SARIMAX': SARIMAXPredictor(order=(1,1,1), seasonal_order=(1,1,1,12), exog_names=selected),
    'VAR': VARPredictor(maxlags=None)
}

# SARIMA (univariado)
models['SARIMA'].fit(train['GDP'])

# SARIMAX (com exógenas)
models['SARIMAX'].fit(train['GDP'], exog=train[selected])

# VAR (multivariado)
models['VAR'].fit(train[['GDP'] + selected[:3]])

# Comparar previsões
for name, model in models.items():
    if name == 'VAR':
        forecast = model.predict_single_variable('GDP', steps=len(test))
    elif name == 'SARIMAX':
        forecast = model.predict(steps=len(test), exog=test[selected])
    else:
        forecast = model.predict(steps=len(test))

    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    print(f"{name}: MAPE = {mape:.2f}%")
```

### Exemplo 3: Análise Exploratória

```python
from src.validation import StationarityTests, GrangerCausality

# Testar estacionaridade de todas as variáveis
tester = StationarityTests()
for col in data.columns:
    print(f"\n{'='*80}")
    print(f"Testando: {col}")
    print('='*80)
    results = tester.run_all_tests(data[col], verbose=True)

# Testar todas as relações causais
gc = GrangerCausality(max_lag=12)
results = gc.test_all_combinations(data, verbose=True)

# Visualizar rede causal
gc.plot_causal_network(threshold=0.05)
```

---

## 📊 Interpretação dos Resultados

### Níveis de Significância

| P-value   | Conclusão                          | Símbolo |
|-----------|-------------------------------------|---------|
| < 0.001   | Forte evidência (99.9%)            | ***     |
| < 0.01    | Evidência significativa (99%)      | **      |
| < 0.05    | Evidência (95%)                    | *       |
| < 0.10    | Evidência fraca (90%)              | .       |
| ≥ 0.10    | Sem evidência                      |         |

### Força de Causalidade de Granger

| P-value     | Força         | Descrição                                    |
|-------------|---------------|----------------------------------------------|
| < 0.001     | very_strong   | Relação causal muito forte                   |
| 0.001-0.01  | strong        | Relação causal forte                         |
| 0.01-0.05   | moderate      | Relação causal moderada                      |
| 0.05-0.10   | weak          | Relação causal fraca                         |
| ≥ 0.10      | none          | Sem relação causal                           |

### Consenso de Estacionaridade

| Votos | Decisão              | Ação Recomendada                  |
|-------|----------------------|-----------------------------------|
| 3/3   | Estacionária         | Usar diretamente                  |
| 2/3   | Provavelmente est.   | Usar, mas monitorar               |
| 1/3   | Provavelmente não    | Aplicar diferenciação             |
| 0/3   | Não estacionária     | Aplicar diferenciação obrigatório |

---

## 🔗 Integração com RL

Após validar preditores, integre com o agente RL:

```python
from src.validation import VariableValidator
from src.models import SARIMAXPredictor, VARPredictor, EnsemblePredictor
from src.agents import AdvancedRLAgent
from src.training import AdvancedRLTrainer

# 1. Validar preditores
validator = VariableValidator()
results = validator.validate_all(data, target_var='GDP')
selected = [p['variable'] for p in results['selected_predictors']]

# 2. Criar modelos com preditores validados
models = [
    SARIMAXPredictor(order=(1,1,1), seasonal_order=(1,1,1,12), exog_names=selected),
    VARPredictor(maxlags=None),
    # ... outros modelos
]

# Treinar modelos
for model in models:
    if isinstance(model, SARIMAXPredictor):
        model.fit(train['GDP'], exog=train[selected])
    elif isinstance(model, VARPredictor):
        model.fit(train[['GDP'] + selected])
    else:
        model.fit(train['GDP'])

# 3. Criar ensemble
ensemble = EnsemblePredictor(models)

# 4. Treinar agente RL para otimizar pesos do ensemble
env = TimeSeriesEnv(data=train, forecast_horizon=12)
agent = AdvancedRLAgent(state_dim=env.observation_space.shape[0])
trainer = AdvancedRLTrainer(env, agent, ensemble)

history = trainer.train(n_episodes=200)
```

---

## 📚 Referências

### Testes de Estacionaridade

1. **Dickey, D. A., & Fuller, W. A. (1979).** Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366a), 427-431.

2. **Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).** Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1-3), 159-178.

3. **Phillips, P. C., & Perron, P. (1988).** Testing for a unit root in time series regression. *Biometrika*, 75(2), 335-346.

### Causalidade de Granger

4. **Granger, C. W. (1969).** Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.

5. **Toda, H. Y., & Yamamoto, T. (1995).** Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics*, 66(1-2), 225-250.

### Modelos de Séries Temporais

6. **Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).** *Time series analysis: forecasting and control* (5th ed.). John Wiley & Sons.

7. **Lütkepohl, H. (2005).** *New introduction to multiple time series analysis*. Springer Science & Business Media.

8. **Hamilton, J. D. (1994).** *Time series analysis* (Vol. 2). Princeton university press.

---

## 🆘 Troubleshooting

### Problema: "Nenhum preditor válido encontrado"

**Possíveis causas:**
1. Dados insuficientes (< 50 observações)
2. Nível de significância muito rigoroso
3. Variáveis não têm relação causal real

**Soluções:**
```python
# 1. Reduzir nível de significância
validator = VariableValidator(significance_level=0.10)  # 10% em vez de 5%

# 2. Aceitar causalidade mais fraca
validator = VariableValidator(min_causal_strength='weak')

# 3. Aumentar max_lag
validator = VariableValidator(max_lag_granger=24)  # Testar mais lags
```

### Problema: "Série não se tornou estacionária após diferenciação"

**Solução:**
```python
# Testar transformações alternativas
import numpy as np

# Log transform
data_log = np.log(data)

# Box-Cox transform
from scipy.stats import boxcox
data_bc, lambda_ = boxcox(data)

# Depois testar estacionaridade
tester.run_all_tests(data_bc)
```

### Problema: "VAR falha com erro de colinearidade"

**Causas:** Variáveis altamente correlacionadas

**Solução:**
```python
# Remover variáveis correlacionadas
correlation = data[var_variables].corr()
print(correlation)

# Manter apenas variáveis com correlação < 0.95
```

---

## 📞 Suporte

Para dúvidas sobre validação de variáveis:
1. Consulte exemplos em `examples/test_validation_advanced.py`
2. Leia documentação dos testes individuais
3. Veja troubleshooting acima

---

**Autor:** Advanced RL Framework
**Nível:** PhD
**Última atualização:** 2025
