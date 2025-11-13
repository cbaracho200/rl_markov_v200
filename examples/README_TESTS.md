# Guia de Testes Avançados

Este diretório contém 3 exemplos completos de testes do framework, em ordem crescente de complexidade.

---

## 📋 Arquivos de Teste

### 1. `test_intermediate.py` ⭐
**Nível:** Intermediário
**Tempo:** 5-10 minutos
**Descrição:** Testa os 4 modelos avançados básicos + RL

**O que testa:**
- ✅ AutoARIMA
- ✅ Prophet
- ✅ CatBoost
- ✅ LightGBM
- ✅ Ensemble
- ✅ Agente RL padrão (PPO)
- ✅ 4 visualizações

**Como executar:**
```bash
cd examples
python test_intermediate.py
```

**Saída esperada:**
- Dados gerados: 250 observações
- 4 modelos treinados
- Comparação de MAPE/RMSE
- 4 gráficos salvos

---

### 2. `test_advanced.py` ⭐⭐
**Nível:** Avançado
**Tempo:** 15-20 minutos
**Descrição:** Teste completo com otimização Bayesiana

**O que testa:**
- ✅ 6 modelos (incluindo LSTM)
- ✅ Otimização com Optuna (30 trials)
- ✅ Agente RL avançado (Transformer)
- ✅ Otimização recursiva
- ✅ Curriculum learning
- ✅ 6 visualizações

**Como executar:**
```bash
cd examples
python test_advanced.py
```

**Saída esperada:**
- Dados: 300 observações
- Otimização: ~30 minutos (Optuna)
- Modelos: 6 treinados
- RL: 150 episódios
- 6 gráficos detalhados

---

### 3. `test_validation_advanced.py` ⭐⭐⭐
**Nível:** PhD
**Tempo:** 5-10 minutos
**Descrição:** Pipeline completo de validação estatística

**O que testa:**
- ✅ Testes de estacionaridade (ADF, KPSS, Phillips-Perron)
- ✅ Testes de causalidade de Granger
- ✅ Seleção automática de preditores
- ✅ SARIMA (sazonal)
- ✅ SARIMAX (com variáveis exógenas)
- ✅ VAR (multivariado)
- ✅ 2 visualizações

**Como executar:**
```bash
cd examples
python test_validation_advanced.py
```

**Saída esperada:**
- Dados: 300 observações econômicas (9 variáveis)
- Validação: 6-8 preditores identificados
- Modelos: SARIMA, SARIMAX, VAR
- Comparação de desempenho
- Gráfico salvo

---

### 4. `test_complete_advanced.py` ⭐⭐⭐⭐ **NOVO!**
**Nível:** PhD+
**Tempo:** 15-20 minutos
**Descrição:** **TESTE COMPLETO DE TODAS AS FUNCIONALIDADES**

**O que testa:**
1. **Validação de Variáveis:**
   - ✅ Testes de estacionaridade (ADF, KPSS, Phillips-Perron)
   - ✅ Testes de causalidade de Granger
   - ✅ Seleção automática de preditores

2. **Modelos Avançados:**
   - ✅ SARIMA
   - ✅ SARIMAX (com exógenas)
   - ✅ VAR (multivariado)
   - ✅ AutoARIMA
   - ✅ Prophet
   - ✅ CatBoost
   - ✅ LightGBM

3. **Dados Complexos:**
   - ✅ 9 variáveis econômicas inter-relacionadas
   - ✅ 400 observações (~33 anos)
   - ✅ Relações causais realistas
   - ✅ Teste negativo (Random_Noise)

**Como executar:**
```bash
cd examples
python test_complete_advanced.py
```

**Saída esperada:**
```
================================================================================
                TESTE COMPLETO DE FUNCIONALIDADES AVANÇADAS
================================================================================

ETAPA 1: GERAÇÃO DE DADOS ECONÔMICOS
  ✓ Dados gerados: 400 observações (~33 anos mensais)
  ✓ Variáveis: 9

ETAPA 2: TESTES DE ESTACIONARIDADE
  ✓ Resumo dos testes de estacionaridade:
  Variable    Stationary  Consensus  ADF_p   KPSS_p   PP_p
  GDP         ✗           1/3        0.1234  0.0100   0.2345
  Interest_Rate ✓         3/3        0.0001  0.1000   0.0002
  ...

ETAPA 3: TESTES DE CAUSALIDADE DE GRANGER
  ✓ Resumo dos testes de Granger:
  Predictor         Causes_GDP  P_value    Best_Lag  Strength
  Interest_Rate     ✓           0.000234   1         very_strong
  Inflation         ✓           0.003456   2         strong
  Random_Noise      ✗           0.734521   3         none
  ...

ETAPA 4: VALIDAÇÃO INTEGRADA DE VARIÁVEIS
  ✓ Validação concluída!
  Total de candidatos: 8
  Preditores selecionados: 6-7

ETAPA 5: TREINAMENTO DE MODELOS AVANÇADOS
  Modelo 1: SARIMA
    ✓ Treinado com sucesso!
    MAPE: 5.67%

  Modelo 2: SARIMAX (com variáveis exógenas)
    Usando 4 preditores: ['Interest_Rate', 'Inflation', ...]
    ✓ Treinado com sucesso!
    MAPE: 4.23%

  ... (7 modelos no total)

ETAPA 6: COMPARAÇÃO DE DESEMPENHO
  🏆 MELHOR MODELO: SARIMAX
     MAPE: 4.23%

RESUMO FINAL
  ✓ Teste completo finalizado!
  Duração: 845.3 segundos (14.1 minutos)

  ✓ Modelos treinados: 7
  🏆 Top 3:
    1. SARIMAX: MAPE = 4.23%
    2. VAR: MAPE = 4.89%
    3. LightGBM: MAPE = 5.12%
```

**Arquivos gerados:**
- `test_results_complete.txt` - Resumo dos resultados

---

## 🎯 Qual teste executar?

| Objetivo | Arquivo | Tempo | Nível |
|----------|---------|-------|-------|
| Teste rápido básico | `test_intermediate.py` | 5-10 min | ⭐ |
| Teste completo com otimização | `test_advanced.py` | 15-20 min | ⭐⭐ |
| Validação estatística | `test_validation_advanced.py` | 5-10 min | ⭐⭐⭐ |
| **Teste completo de TUDO** | **`test_complete_advanced.py`** | **15-20 min** | **⭐⭐⭐⭐** |

---

## 📊 Comparação dos Testes

| Feature | Intermediate | Advanced | Validation | **Complete** |
|---------|-------------|----------|------------|--------------|
| Modelos básicos | ✅ (4) | ✅ (6) | ❌ | ✅ (7) |
| SARIMA/SARIMAX/VAR | ❌ | ❌ | ✅ | ✅ |
| Testes de estacionaridade | ❌ | ❌ | ✅ | ✅ |
| Testes de Granger | ❌ | ❌ | ✅ | ✅ |
| Validação integrada | ❌ | ❌ | ✅ | ✅ |
| Otimização Optuna | ❌ | ✅ | ❌ | ⚠️ (opcional) |
| Agente RL | ✅ Padrão | ✅ Avançado | ❌ | ⚠️ (não incluído) |
| Visualizações | 4 | 6 | 2 | ❌ |

✅ = Incluído
⚠️ = Opcional/Desabilitado
❌ = Não incluído

---

## 🔧 Troubleshooting

### Erro: ModuleNotFoundError
```bash
# Certifique-se de estar no diretório correto
cd /caminho/para/Previs-o-ciclos-Econ-mico
python examples/test_complete_advanced.py

# Ou adicione ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/caminho/para/Previs-o-ciclos-Econ-mico"
```

### Erro: Memória insuficiente
```python
# Reduza o tamanho dos dados em test_complete_advanced.py:
data = generate_advanced_economic_data(n=200)  # Em vez de 400
```

### Teste muito lento
```python
# Desative otimização Bayesiana (já está desabilitada por padrão)
# Reduza número de modelos comentando alguns
```

---

## 📝 Personalizando os Testes

### Adicionar novos dados
```python
# Em qualquer arquivo de teste, substitua:
data = generate_synthetic_data(...)

# Por:
import pandas as pd
data = pd.read_csv('seus_dados.csv')
```

### Ajustar hiperparâmetros
```python
# Exemplo: Aumentar iterações do CatBoost
catboost = CatBoostPredictor(
    iterations=500,  # Padrão: 200
    learning_rate=0.03  # Padrão: 0.05
)
```

### Mudar horizonte de previsão
```python
# Padrão: 12 períodos
forecast = model.predict(steps=24)  # 24 períodos
```

---

## 🎓 Próximos Passos

Após executar os testes:

1. **Entender os resultados:**
   - Compare MAPE entre modelos
   - Analise quais preditores foram selecionados
   - Verifique critérios de informação (AIC, BIC)

2. **Usar em dados reais:**
   - Carregue seus próprios dados
   - Execute validação de variáveis
   - Treine modelos com preditores validados

3. **Otimizar ainda mais:**
   - Ative otimização Bayesiana
   - Ajuste hiperparâmetros manualmente
   - Teste diferentes combinações de modelos

4. **Integrar com RL:**
   - Use ensemble otimizado por RL
   - Treine agente avançado com Transformer
   - Implemente otimização recursiva

---

## 📚 Documentação Relacionada

- **[VALIDATION_GUIDE.md](../VALIDATION_GUIDE.md)** - Guia completo de validação
- **[ADVANCED_MODELS.md](../ADVANCED_MODELS.md)** - Detalhes de cada modelo
- **[QUICK_START.md](../QUICK_START.md)** - Como começar
- **[README.md](../README.md)** - Visão geral do framework

---

## 🆘 Suporte

Se encontrar problemas:
1. Verifique se todas as dependências estão instaladas: `pip install -r requirements.txt`
2. Consulte o [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
3. Leia os comentários no código de cada teste
4. Abra uma issue no GitHub

---

**Framework:** Advanced RL for Economic Forecasting v2.1
**Última atualização:** 2025
**Nível:** PhD+
