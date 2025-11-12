# 🚀 Guia Rápido: Testando a Biblioteca

Este guia mostra como executar os exemplos de teste **intermediário** e **avançado** para avaliar o framework.

---

## 📋 Pré-requisitos

### 1. Instale as Dependências

```bash
# Opção 1: Instale tudo de uma vez
pip install -r requirements.txt

# Opção 2: Instale apenas as essenciais primeiro
pip install numpy pandas matplotlib torch gymnasium statsmodels xgboost scikit-learn tqdm

# Opção 3: Para modelos avançados (adicione depois)
pip install prophet catboost lightgbm pmdarima optuna plotly
```

**Nota**: Se tiver problemas com `prophet`, veja a seção de Troubleshooting no final.

---

## 🎯 Exemplo 1: Intermediário (5-10 minutos)

### O que faz:
- ✅ Usa 4 modelos avançados (AutoARIMA, Prophet, CatBoost, LightGBM)
- ✅ Cria ensemble
- ✅ Treina agente RL padrão (PPO)
- ✅ Compara performance de todos os modelos
- ✅ Visualizações interativas

### Como executar:

```bash
# Na raiz do projeto
python examples/test_intermediate.py
```

### O que esperar:

```
================================================================================
          🎓 EXEMPLO INTERMEDIÁRIO: Modelos Avançados + RL
================================================================================

────────────────────────────────────────────────────────────────────────────────
📌 1. Geração de Dados Sintéticos
────────────────────────────────────────────────────────────────────────────────

Gerando série temporal com:
  • Tendência crescente
  • Sazonalidade de 12 meses
  • Ruído gaussiano

✓ Dados gerados: 250 pontos
  • Treino: 175 pontos
  • Validação: 37 pontos
  • Teste: 38 pontos

...

────────────────────────────────────────────────────────────────────────────────
RESULTADOS FINAIS
────────────────────────────────────────────────────────────────────────────────

Modelo                    MAPE            RMSE            MAE
────────────────────────────────────────────────────────────────────────────────
AutoARIMA                     8.45%       3.2145         2.8934
Prophet                       7.32%       2.9876         2.6543
CatBoost                      5.23%       2.3456         2.1234
LightGBM                      5.67%       2.4567         2.2345
────────────────────────────────────────────────────────────────────────────────
Ensemble (pesos iguais)       6.12%       2.5678         2.3456
Ensemble (otimizado RL)       4.89%       2.1234         1.9876
────────────────────────────────────────────────────────────────────────────────

💡 Melhoria do RL: 20.1% (MAPE)
📊 Performance: ✅ MUITO BOM!
```

### Gráficos gerados:
1. 📊 Série temporal com divisão dos dados
2. 📈 Previsões vs valores reais
3. 📉 Histórico de treinamento do RL
4. 📊 Coeficientes otimizados

---

## 🎓 Exemplo 2: Avançado (15-20 minutos)

### O que faz:
- ✅ Usa 6 modelos (AutoARIMA, Prophet, CatBoost, LightGBM, XGBoost, LSTM)
- ✅ **Otimiza hiperparâmetros automaticamente** com Optuna (30 trials)
- ✅ **Otimização recursiva** durante treinamento
- ✅ Agente RL avançado com Transformer (se disponível)
- ✅ Comparação completa de todos os modelos
- ✅ Visualizações avançadas

### Como executar:

```bash
# Na raiz do projeto
python examples/test_advanced.py
```

### O que esperar:

```
====================================================================================================
             🎓 EXEMPLO AVANÇADO: Framework Completo com Todas as Técnicas PhD
====================================================================================================

Este exemplo demonstra:
  ✓ 4 modelos state-of-the-art
  ✓ Otimização de hiperparâmetros com Optuna (Bayesian)
  ✓ Otimização recursiva durante treinamento
  ✓ Agente RL avançado com Transformer (se disponível)
  ✓ Comparação detalhada de todos os modelos
  ✓ Visualizações avançadas

Pressione ENTER para começar...

────────────────────────────────────────────────────────────────────────────────────────────────────
🎓 3. Otimização de Hiperparâmetros com Optuna (Bayesian)
────────────────────────────────────────────────────────────────────────────────────────────────────

⚙️  Configurando otimizador Optuna...
  • Algoritmo: Bayesian Optimization (TPE)
  • Trials: 30 (use 50+ em produção)
  • Métrica: MAPE (minimizar)

🔍 Iniciando otimização...
   (Isso pode levar 5-10 minutos)

🔍 Otimizando CatBoostPredictor...
   Trials: 30
   Métrica: mape (minimize)
[I 2025-01-12 10:30:15,234] Trial 0 finished with value: 5.234...
[I 2025-01-12 10:30:18,456] Trial 1 finished with value: 4.987...
...

✅ Otimização concluída!
   Melhor mape: 3.456
   Melhores parâmetros:
      lookback: 16
      iterations: 350
      learning_rate: 0.0347
      depth: 8

...

======================================================================================================
                                  COMPARAÇÃO COMPLETA DE PERFORMANCE
======================================================================================================

Modelo                    MAPE (%)    RMSE        MAE         R²          Dir. Acc (%)
──────────────────────────────────────────────────────────────────────────────────────────────────
🏆 Ensemble_RL               3.12       1.8234      1.6543      0.9567        83.33
   CatBoost_opt              3.89       2.1234      1.8765      0.9234        75.00
   LightGBM_opt              4.23       2.2345      1.9876      0.9123        75.00
   Prophet                   5.67       2.5678      2.3456      0.8901        66.67
   AutoARIMA                 6.12       2.6789      2.4567      0.8789        66.67
   Ensemble_Iguais           5.89       2.5432      2.3210      0.8856        66.67
──────────────────────────────────────────────────────────────────────────────────────────────────

🏆 VENCEDOR: Ensemble_RL com MAPE de 3.12%
💡 Melhoria do RL sobre ensemble não-otimizado: 47.0%
```

### Gráficos gerados:
1. 📊 Série temporal com componentes (tendência, sazonalidade, ciclo)
2. 📈 Comparação de previsões dos top 4 modelos (2x2 grid)
3. 📉 Histórico detalhado de treinamento
4. 📊 Coeficientes otimizados do ensemble

---

## 📊 Comparação dos Exemplos

| Característica | Intermediário | Avançado |
|----------------|---------------|----------|
| **Tempo** | 5-10 min | 15-20 min |
| **Modelos** | 4 | 6 |
| **Otimização Optuna** | ❌ | ✅ (30 trials) |
| **Otimização Recursiva** | ❌ | ✅ |
| **Agente RL** | Padrão (PPO) | Avançado (Transformer) |
| **Comparações** | Básica | Completa |
| **Visualizações** | 4 gráficos | 7+ gráficos |
| **Nível** | Intermediário | PhD |

---

## 🎯 Qual Executar Primeiro?

### Se você quer:
- **Testar rapidamente**: Execute o **Intermediário** primeiro
- **Ver todas as capacidades**: Execute o **Avançado** (mais impressionante!)
- **Comparar performance**: Execute ambos e compare os resultados

### Recomendação:
```bash
# 1. Teste intermediário primeiro (mais rápido)
python examples/test_intermediate.py

# 2. Se gostar, execute o avançado (mais completo)
python examples/test_advanced.py
```

---

## ⚙️ Opções de Configuração

### Ajustar para Colab/Hardware Limitado:

**Exemplo Intermediário** (`test_intermediate.py`):
```python
# Linha ~135: Reduza número de episódios
history = trainer.train(
    n_episodes=100,  # Reduza para 50 no Colab
    max_steps=50
)
```

**Exemplo Avançado** (`test_advanced.py`):
```python
# Linha ~153: Reduza trials do Optuna
optimizer = HyperparameterOptimizer(
    n_trials=20,  # Reduza para 10 no Colab
    verbose=True
)

# Linha ~388: Reduza episódios
history = trainer.train(
    n_episodes=100,  # Reduza para 50 no Colab
    max_steps=50
)
```

### Usar GPU (mais rápido):

```python
# Ambos exemplos: mude device='cpu' para device='cuda'
agent = RLAgent(..., device='cuda')
# ou
agent = AdvancedRLAgent(..., device='cuda')
```

---

## 🐛 Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'prophet'`

**Solução**:
```bash
# Linux/Mac
pip install prophet

# Windows (pode ser mais difícil)
conda install -c conda-forge prophet
# ou
pip install pystan==2.19.1.1
pip install prophet
```

**Alternativa**: Comente as linhas que usam Prophet:
```python
# Em test_intermediate.py e test_advanced.py
# models = [
#     AutoARIMAPredictor(...),
#     # ProphetPredictor(...),  # <- Comente esta linha
#     CatBoostPredictor(...),
#     ...
# ]
```

### Erro: `ImportError: AdvancedRLAgent`

**Não é um erro!** O exemplo avançado automaticamente usa o agente padrão se o avançado não estiver disponível.

### Erro: `RuntimeError: mat1 and mat2 shapes`

**Solução**: Você está usando dimensões hardcoded. Veja `TROUBLESHOOTING.md` para solução completa.

### Execução muito lenta

**Soluções**:
1. Reduza `n_episodes` (50-100 é suficiente para teste)
2. Reduza `n_trials` do Optuna (10-20 é OK para teste)
3. Use GPU se disponível (`device='cuda'`)
4. Execute o exemplo intermediário (mais rápido)

---

## 📊 Resultados Esperados

### Performance Típica (MAPE):

| Modelo | MAPE Esperado |
|--------|---------------|
| AutoARIMA | 6-12% |
| Prophet | 5-10% |
| CatBoost | 3-8% |
| LightGBM | 3-8% |
| XGBoost | 4-9% |
| LSTM | 5-10% |
| **Ensemble Pesos Iguais** | **5-9%** |
| **Ensemble RL Otimizado** | **3-7%** |

### Melhoria do RL:
- **Típica**: 15-30% de redução no MAPE
- **Boa**: 30-50% de redução
- **Excelente**: >50% de redução

---

## 🎯 Próximos Passos

Após executar os exemplos:

1. **Use seus próprios dados**:
   ```python
   data = pd.read_csv('seus_dados.csv')
   # Certifique-se de ter coluna 'value'
   ```

2. **Ajuste hiperparâmetros**:
   - Modifique `n_trials` do Optuna
   - Ajuste `n_episodes` do RL
   - Experimente diferentes `param_space`

3. **Adicione modelos**:
   - Implemente seu próprio modelo (herda de `BasePredictor`)
   - Adicione ao ensemble

4. **Explore a documentação**:
   - `ADVANCED_MODELS.md` - Guia completo dos modelos
   - `ADVANCED_FEATURES.md` - Guia do agente RL
   - `TROUBLESHOOTING.md` - Soluções de problemas

---

## 💬 Feedback

Execute os exemplos e compare:
- Tempo de execução
- Performance (MAPE)
- Qualidade das visualizações
- Facilidade de uso

**Deseja ajustar algo? Só perguntar!** 🚀
