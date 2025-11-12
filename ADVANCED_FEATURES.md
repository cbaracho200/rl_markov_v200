# 🎓 Agente RL Avançado - Nível PhD

## Visão Geral

Este framework implementa um agente de Reinforcement Learning de **nível PhD** com técnicas state-of-the-art para previsão de séries temporais econômicas. O agente é baseado em **Proximal Policy Optimization (PPO)** com múltiplas melhorias arquiteturais e algorítmicas.

---

## 🚀 Técnicas Implementadas

### 1. **Transformer-based Actor-Critic**
- **Descrição**: Arquitetura baseada em Transformer substituindo redes feedforward tradicionais
- **Benefícios**:
  - Captura dependências de longo alcance em séries temporais
  - Processamento paralelo eficiente
  - Melhor representação de padrões complexos
- **Implementação**: `TransformerActorCritic` em `rl_agent_advanced.py`

### 2. **Multi-Head Self-Attention**
- **Descrição**: Mecanismo de atenção com múltiplas cabeças para capturar diferentes aspectos dos dados
- **Parâmetros**: 8 cabeças de atenção por padrão
- **Benefícios**:
  - Foca em diferentes partes da sequência temporal simultaneamente
  - Aprende representações hierárquicas
  - Melhora interpretabilidade

### 3. **Prioritized Experience Replay (PER)**
- **Descrição**: Buffer de replay que prioriza transições com maior TD-error
- **Estrutura**: Sum Tree para sampling eficiente O(log n)
- **Benefícios**:
  - Aprende mais rápido com experiências "surpresa"
  - Uso mais eficiente de dados
  - Convergência mais estável
- **Parâmetros**:
  - α = 0.6 (priorização)
  - β = 0.4 → 1.0 (importance sampling)

### 4. **Noisy Networks**
- **Descrição**: Adiciona ruído parametrizado aos pesos da rede
- **Benefícios**:
  - Exploração adaptativa sem epsilon-greedy
  - Ruído diminui naturalmente com o treinamento
  - Mais eficiente que exploração uniforme
- **Implementação**: `NoisyLinear` substituindo `nn.Linear`

### 5. **Dueling Architecture**
- **Descrição**: Separa função valor em advantage e value streams
- **Fórmula**: `Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))`
- **Benefícios**:
  - Aprende melhor quais estados são valiosos
  - Melhora generalização
  - Convergência mais rápida

### 6. **LSTM Memory**
- **Descrição**: Camadas recorrentes para memória de longo prazo
- **Configuração**: 2 camadas LSTM após Transformer
- **Benefícios**:
  - Mantém contexto temporal entre steps
  - Captura dependências de prazo muito longo
  - Complementa atenção do Transformer

### 7. **Ensemble de Critics (3 Critics)**
- **Descrição**: Três redes critic independentes; usa média das previsões
- **Benefícios**:
  - Reduz viés de estimativa
  - Maior robustez
  - Menor variância nas estimativas de valor

### 8. **Adaptive Entropy Regularization**
- **Descrição**: Coeficiente de entropia ajustável automaticamente
- **Método**: Otimização dual com target entropy
- **Benefícios**:
  - Balanceamento automático exploração/exploração
  - Adapta-se à fase do treinamento
  - Não requer tuning manual

### 9. **Learning Rate Scheduling com Warmup**
- **Estratégia**: Cosine Annealing with Warm Restarts
- **Configuração**:
  - Warmup: 1000 steps
  - T_0 = 10, T_mult = 2
  - eta_min = lr * 0.1
- **Benefícios**:
  - Convergência mais estável
  - Evita mínimos locais
  - Melhora generalização

### 10. **Gradient Accumulation**
- **Descrição**: Suporta mini-batches para simular batches grandes
- **Benefícios**:
  - Treina com batches grandes em hardware limitado
  - Estimativas de gradiente mais estáveis

### 11. **Spectral Normalization**
- **Descrição**: Normaliza pesos para controlar Lipschitz constant
- **Benefícios**:
  - Estabilidade no treinamento
  - Previne explosão/desaparecimento de gradientes

### 12. **Curriculum Learning**
- **Descrição**: Aumenta dificuldade progressivamente
- **Estágios**:
  - Easy (0-30%): Padrões simples
  - Medium (30-60%): Complexidade moderada
  - Hard (60-100%): Cenários completos
- **Benefícios**:
  - Aprendizado mais eficiente
  - Menos falhas catastróficas

### 13. **Early Stopping**
- **Descrição**: Para treinamento se não há melhora
- **Patience**: 50 avaliações
- **Benefícios**:
  - Previne overfitting
  - Economiza tempo computacional

### 14. **Value Function Clipping**
- **Descrição**: Clipa atualizações do critic como no actor
- **Benefícios**:
  - Atualizações mais conservadoras
  - Maior estabilidade

### 15. **Gradient Clipping**
- **Método**: Clip por norma (max_norm = 0.5)
- **Benefícios**:
  - Previne explosão de gradientes
  - Treinamento mais estável

---

## 📊 Comparação: Standard vs Advanced

| Característica | Standard RL | Advanced RL (PhD) |
|---|---|---|
| Arquitetura | Feedforward | Transformer + LSTM |
| Atenção | Nenhuma | Multi-Head (8 heads) |
| Replay Buffer | FIFO simples | Prioritized (PER) |
| Exploração | Ruído gaussiano | Noisy Networks |
| Critics | 1 critic | Ensemble de 3 |
| Entropy Coef | Fixo | Adaptativo |
| Learning Rate | Fixo | Scheduling + Warmup |
| Curriculum | Não | Sim (3 estágios) |
| Early Stop | Não | Sim (patience=50) |
| Parâmetros | ~50K | ~500K |

---

## 🎯 Quando Usar Cada Versão

### Use **Standard RL** (`RLAgent`) quando:
- ✅ Dados limitados (< 100 pontos)
- ✅ Hardware limitado (CPU básico)
- ✅ Prototipagem rápida
- ✅ Séries simples com poucos padrões

### Use **Advanced RL** (`AdvancedRLAgent`) quando:
- ✅ Dados abundantes (> 200 pontos)
- ✅ Séries complexas com múltiplos padrões
- ✅ GPU disponível (recomendado)
- ✅ Máxima precisão necessária
- ✅ Produção / Research

---

## 💻 Como Usar

### Exemplo Básico

```python
from src.agents import AdvancedRLAgent
from src.training import AdvancedRLTrainer
from src.environments import TimeSeriesEnv

# 1. Cria ambiente
env = TimeSeriesEnv(data=your_data, forecast_horizon=12)

# 2. Cria agente avançado
agent = AdvancedRLAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dim=512,
    num_heads=8,
    num_layers=3,
    use_per=True,
    use_noisy=True,
    use_lstm=True,
    device='cuda'  # Ou 'cpu'
)

# 3. Cria trainer
trainer = AdvancedRLTrainer(
    env=env,
    agent=agent,
    use_curriculum=True
)

# 4. Treina
history = trainer.train(
    n_episodes=500,
    early_stopping=True
)

# 5. Avalia
results = trainer.evaluate(n_episodes=10)
```

### Exemplo Completo

Veja `examples/advanced_rl_example.py` para exemplo completo com:
- Geração de dados sintéticos
- Configuração detalhada
- Visualizações
- Métricas avançadas

---

## 🔧 Hiperparâmetros Recomendados

### Para GPU (RTX 3090+)
```python
agent = AdvancedRLAgent(
    hidden_dim=512,
    num_heads=8,
    num_layers=3,
    buffer_size=100000,
    learning_rate=1e-4,
    device='cuda'
)

trainer.train(
    n_episodes=1000,
    batch_size=128
)
```

### Para CPU / Google Colab Free
```python
agent = AdvancedRLAgent(
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    buffer_size=50000,
    learning_rate=3e-4,
    device='cpu'
)

trainer.train(
    n_episodes=200,
    batch_size=64
)
```

---

## 📈 Métricas de Treinamento

O trainer avançado rastreia:
- **Episode Rewards**: Recompensa por episódio
- **Policy Loss**: Perda do actor
- **Value Loss**: Perda do critic
- **Entropy**: Nível de exploração
- **Entropy Coefficient**: Coef adaptativo
- **Learning Rate**: LR ao longo do tempo
- **Gradient Norms**: Para debug
- **Buffer Size**: Tamanho do replay buffer

---

## 🎓 Referências Acadêmicas

1. **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
2. **Transformers**: Vaswani et al. (2017) - "Attention Is All You Need"
3. **PER**: Schaul et al. (2016) - "Prioritized Experience Replay"
4. **Noisy Nets**: Fortunato et al. (2018) - "Noisy Networks for Exploration"
5. **Dueling**: Wang et al. (2016) - "Dueling Network Architectures"
6. **GAE**: Schulman et al. (2016) - "High-Dimensional Continuous Control"
7. **SAC (Entropy)**: Haarnoja et al. (2018) - "Soft Actor-Critic"

---

## 🐛 Debugging

### Problema: Loss não converge
**Solução**: Reduza learning rate, aumente warmup steps

### Problema: Exploração excessiva
**Solução**: Ajuste target_entropy para valor mais negativo

### Problema: OOM (Out of Memory)
**Solução**: Reduza hidden_dim, num_layers, ou batch_size

### Problema: Treinamento muito lento
**Solução**: Use GPU, reduza buffer_size, ou use agente standard

---

## 📞 Suporte

Para questões sobre implementação ou bugs, abra uma issue no repositório.

---

## 📄 Licença

Este código é fornecido para fins educacionais e de pesquisa.
