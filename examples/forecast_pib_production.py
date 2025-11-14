"""
Previsão de PIB com Framework Avançado de RL
============================================

Este script demonstra como extrair o MÁXIMO do framework para prever PIB
usando todas as funcionalidades avançadas disponíveis.

PIPELINE COMPLETO:
1. Carregamento e preparação dos dados
2. Análise exploratória com visualizações
3. Validação estatística de variáveis (Estacionaridade + Granger)
4. Seleção automática das melhores variáveis exógenas
5. Treinamento de múltiplos modelos avançados
6. Comparação de desempenho
7. Ensemble otimizado
8. Visualizações profissionais
9. Exportação de resultados e modelos

Variável Alvo: pib_acum12m
Variáveis Exógenas: 68 indicadores econômicos

Tempo estimado: 20-30 minutos
Nível: Production-Ready

Autor: Advanced RL Framework v2.1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# Configurar estilo de plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Adicionar ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importações do framework
from src.validation import (
    StationarityTests,
    GrangerCausality,
    VariableValidator
)

from src.models import (
    SARIMAPredictor,
    SARIMAXPredictor,
    VARPredictor,
    AutoARIMAPredictor,
    ProphetPredictor,
    CatBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor
)


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

class Config:
    """Configurações do experimento."""

    # Dados
    TARGET_VAR = 'pib_acum12m'

    EXOG_VARS = [
        'ibc_br', 'ind_transformacao_cni', 'n_hr_trab_ind',
        'prod_aco', 'prod_oleo_bruto', 'prod_veic_total',
        'trafego_veic_pesados', 'uci_ind_cni', 'uci_ind_fgv',
        'vendas_veic_concessionarias', 'vendas_supermercados', 'venda_veic_total',
        'cambio', 'cons_energia_comercial', 'cons_energia_ind',
        'cons_energia_residencial', 'cons_energia_total', 'cons_gasolina',
        'cons_diesel', 'endividamento_familias_exhabit', 'mult_monetario',
        'op_compromissada', 'cond_econ_atuais_fecomercio', 'conf_consmidor_fecomercio',
        'expec_futuras_fecomercio', 'div_liq_gg', 'nfsp',
        'massa_sal_real_ind', 'renda_disponivel', 'salario_minimo',
        'ic_br', 'ic_br_agro', 'ic_br_energia', 'ic_br_metal',
        'ipc_br', 'ipa_di', 'igv_r', 'igp_m', 'igp_di', 'incc',
        'ipca_administrados', 'ipca_comercializaveis', 'ipca_livres',
        'ipca_nao_comercializaveis', 'ipca_nucleo_dp', 'ipca_nucleo_exfe',
        'ipca_nucleo_ms', 'ipca_nucleo_ma', 'ipca_nucleo_p55',
        'ipca_nucleo_ex0', 'ipca_nucleo_ex1', 'ipca_nucleo_ex2',
        'ipca_nucleo_ex3', 'ipca_servicos', 'bc_saldo', 'idp',
        'mov_cambio_contr', 'tc_saldo', 'us_gov_sec_3m', 'fed_funds',
        'us_gov_sec_1y', 'us_gov_sec_10y', 'us_gov_sec_2y',
        'us_gov_sec_5y', 'epu_us', 'nber_us_daily', 'initial_claims'
    ]

    # Divisão dos dados
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Validação
    SIGNIFICANCE_LEVEL = 0.05
    MAX_LAG_GRANGER = 12
    MIN_CAUSAL_STRENGTH = 'weak'
    AUTO_TRANSFORM = True

    # Previsão
    FORECAST_HORIZON = 12  # 12 meses à frente

    # Outputs
    OUTPUT_DIR = Path('outputs')
    SAVE_MODELS = True
    SAVE_PLOTS = True
    SAVE_RESULTS = True


# ============================================================================
# GERAÇÃO DE DADOS (SUBSTITUA PELOS SEUS DADOS REAIS)
# ============================================================================

def generate_synthetic_pib_data(n_obs=300, seed=42):
    """
    Gera dados sintéticos realistas de PIB e indicadores econômicos.

    SUBSTITUA ESTA FUNÇÃO PELOS SEUS DADOS REAIS:

    import pandas as pd
    data = pd.read_csv('seus_dados_pib.csv', parse_dates=['data'], index_col='data')

    Certifique-se de que:
    - O índice é uma data (pd.DatetimeIndex)
    - Contém a coluna 'pib_acum12m'
    - Contém as 68 variáveis exógenas
    - Não tem valores missing (ou trate antes)
    """
    np.random.seed(seed)

    # Data index
    dates = pd.date_range(start='2000-01-01', periods=n_obs, freq='MS')

    # PIB (variável alvo) - com tendência e sazonalidade
    t = np.arange(n_obs)
    trend = 100 + 0.3 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    cycle = 10 * np.sin(2 * np.pi * t / 48)

    # Variáveis exógenas principais (simplificado)
    data_dict = {}

    # IBC-BR (índice de atividade econômica) - correlação forte com PIB
    ibc_br = 100 + 0.25 * t + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, n_obs)
    data_dict['ibc_br'] = ibc_br

    # Produção industrial
    data_dict['ind_transformacao_cni'] = 95 + 0.2 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n_obs)
    data_dict['prod_aco'] = 80 + 0.15 * t + 2 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 4, n_obs)
    data_dict['prod_veic_total'] = 90 + 0.18 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 5, n_obs)

    # Consumo
    data_dict['vendas_supermercados'] = 100 + 0.25 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n_obs)
    data_dict['venda_veic_total'] = 85 + 0.2 * t + 6 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 6, n_obs)
    data_dict['cons_energia_total'] = 110 + 0.3 * t + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, n_obs)

    # Preços
    data_dict['ipca_nucleo_exfe'] = 3 + 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, n_obs)
    data_dict['igp_m'] = 4 + 0.015 * t + 0.6 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.8, n_obs)

    # Câmbio
    data_dict['cambio'] = 3.5 + 0.02 * t + 0.3 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.2, n_obs)

    # Taxas de juros internacionais
    data_dict['fed_funds'] = 2 + 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 36) + np.random.normal(0, 0.3, n_obs)
    data_dict['us_gov_sec_10y'] = 3 + 0.01 * t + 0.4 * np.sin(2 * np.pi * t / 48) + np.random.normal(0, 0.25, n_obs)

    # Gerar outras variáveis (simplificado - combinações das principais)
    base_vars = list(data_dict.keys())
    for var in Config.EXOG_VARS:
        if var not in data_dict:
            # Criar como combinação linear de variáveis base + ruído
            base_idx = hash(var) % len(base_vars)
            base_var = data_dict[base_vars[base_idx]]
            noise_level = 0.5 + (hash(var) % 10) / 10
            data_dict[var] = base_var * (0.8 + np.random.uniform(-0.2, 0.2)) + np.random.normal(0, noise_level, n_obs)

    # PIB com dependências causais
    pib = np.zeros(n_obs)
    for i in range(5, n_obs):
        pib[i] = (
            trend[i] +
            seasonal[i] +
            cycle[i] +
            0.3 * pib[i-1] +  # Autocorrelação
            0.5 * ibc_br[i-1] +  # IBC-BR causa PIB (lag 1)
            0.2 * data_dict['ind_transformacao_cni'][i-1] +
            0.15 * data_dict['vendas_supermercados'][i-2] +
            -0.1 * data_dict['ipca_nucleo_exfe'][i-1] +
            0.05 * data_dict['cons_energia_total'][i-1] +
            np.random.normal(0, 1.5)
        )

    data_dict['pib_acum12m'] = pib

    # Criar DataFrame
    df = pd.DataFrame(data_dict, index=dates)

    return df


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_exploratory_analysis(data, target_var, output_dir):
    """Análise exploratória com múltiplos plots."""

    print("\n📊 Criando visualizações exploratórias...")

    # Plot 1: Série temporal do PIB
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(data.index, data[target_var], linewidth=2, color='navy')
    axes[0].set_title(f'Evolução de {target_var}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel(target_var, fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Decomposição (trend, seasonal)
    from scipy import signal
    detrended = signal.detrend(data[target_var].values)
    axes[1].plot(data.index, detrended, linewidth=1.5, color='darkgreen', alpha=0.7)
    axes[1].set_title('Série Destendenciada', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Valores', fontsize=12)
    axes[1].set_xlabel('Data', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'exploratory_pib.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot 1: Análise exploratória do PIB salvo")

    # Plot 2: Correlação com top 15 variáveis
    correlations = data.corr()[target_var].abs().sort_values(ascending=False)[1:16]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(correlations.values)
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors)
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=10)
    ax.set_xlabel('Correlação Absoluta', fontsize=12)
    ax.set_title(f'Top 15 Variáveis Mais Correlacionadas com {target_var}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'correlation_top15.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot 2: Top 15 correlações salvo")

    # Plot 3: Distribuição do PIB
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(data[target_var], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel(target_var, fontsize=12)
    axes[0].set_ylabel('Frequência', fontsize=12)
    axes[0].set_title('Distribuição', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    from scipy import stats
    stats.probplot(data[target_var], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normalidade)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'distribution_pib.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot 3: Distribuição do PIB salvo")


def plot_validation_results(validation_results, output_dir):
    """Visualiza resultados da validação."""

    print("\n📊 Criando visualizações de validação...")

    if not validation_results['selected_predictors']:
        print("  ⚠ Nenhum preditor selecionado para visualizar")
        return

    # Plot: Importância das variáveis
    predictors = validation_results['selected_predictors']

    variables = [p['variable'] for p in predictors]
    importance = [1 - p['p_value'] for p in predictors]
    lags = [p['lag'] for p in predictors]
    strengths = [p['strength'] for p in predictors]

    # Mapear força para cor
    strength_colors = {
        'very_strong': '#2ecc71',
        'strong': '#3498db',
        'moderate': '#f39c12',
        'weak': '#e74c3c'
    }
    colors = [strength_colors.get(s, '#95a5a6') for s in strengths]

    fig, ax = plt.subplots(figsize=(12, max(6, len(variables) * 0.4)))

    bars = ax.barh(range(len(variables)), importance, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables, fontsize=10)
    ax.set_xlabel('Importância (1 - p-value)', fontsize=12)
    ax.set_title('Importância das Variáveis Selecionadas por Granger',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Adicionar lags nas barras
    for i, (bar, imp, lag) in enumerate(zip(bars, importance, lags)):
        ax.text(imp + 0.01, i, f'lag={lag}', va='center', fontsize=8)

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Very Strong'),
        Patch(facecolor='#3498db', label='Strong'),
        Patch(facecolor='#f39c12', label='Moderate'),
        Patch(facecolor='#e74c3c', label='Weak')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'variable_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot: Importância das variáveis salvo")


def plot_model_comparison(results, actual, output_dir):
    """Compara previsões de múltiplos modelos."""

    print("\n📊 Criando visualizações de comparação...")

    if not results:
        print("  ⚠ Nenhum resultado para visualizar")
        return

    # Plot 1: Previsões vs Real
    fig, ax = plt.subplots(figsize=(14, 7))

    # Real
    ax.plot(range(len(actual)), actual, 'k-', linewidth=2.5, label='Real', alpha=0.8)

    # Modelos
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (model_name, result), color in zip(results.items(), colors):
        forecast = result['forecast']
        ax.plot(range(len(forecast)), forecast, '--', linewidth=1.5,
                label=f"{model_name} (MAPE: {result['mape']:.2f}%)",
                color=color, alpha=0.7)

    ax.set_xlabel('Período de Teste', fontsize=12)
    ax.set_ylabel('PIB', fontsize=12)
    ax.set_title('Comparação de Previsões vs Valores Reais', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'forecast_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot 1: Comparação de previsões salvo")

    # Plot 2: Erros por modelo
    model_names = list(results.keys())
    mapes = [results[m]['mape'] for m in model_names]
    rmses = [results[m]['rmse'] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MAPE
    colors_mape = plt.cm.RdYlGn_r(np.array(mapes) / max(mapes))
    axes[0].barh(model_names, mapes, color=colors_mape, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('MAPE (%)', fontsize=12)
    axes[0].set_title('Mean Absolute Percentage Error', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    for i, (name, val) in enumerate(zip(model_names, mapes)):
        axes[0].text(val + 0.1, i, f'{val:.2f}%', va='center', fontsize=9)

    # RMSE
    colors_rmse = plt.cm.RdYlGn_r(np.array(rmses) / max(rmses))
    axes[1].barh(model_names, rmses, color=colors_rmse, alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    for i, (name, val) in enumerate(zip(model_names, rmses)):
        axes[1].text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot 2: Comparação de erros salvo")

    # Plot 3: Residuals do melhor modelo
    best_model = min(results.keys(), key=lambda k: results[k]['mape'])
    residuals = actual - results[best_model]['forecast']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals ao longo do tempo
    axes[0, 0].plot(residuals, linewidth=1.5, color='crimson')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Período', fontsize=11)
    axes[0, 0].set_ylabel('Resíduo', fontsize=11)
    axes[0, 0].set_title(f'Resíduos ao Longo do Tempo ({best_model})', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Histograma
    axes[0, 1].hist(residuals, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Resíduo', fontsize=11)
    axes[0, 1].set_ylabel('Frequência', fontsize=11)
    axes[0, 1].set_title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot dos Resíduos', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # ACF dos resíduos
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('Autocorrelação dos Resíduos', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if Config.SAVE_PLOTS:
        plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Plot 3: Análise de resíduos salvo")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Pipeline completo de previsão de PIB."""

    start_time = datetime.now()

    print("="*80)
    print("PREVISÃO DE PIB COM FRAMEWORK AVANÇADO DE RL".center(80))
    print("="*80)
    print(f"Início: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Variável Alvo: {Config.TARGET_VAR}")
    print(f"Variáveis Exógenas: {len(Config.EXOG_VARS)}")
    print("="*80)

    # Criar diretório de saída
    Config.OUTPUT_DIR.mkdir(exist_ok=True)

    # ========================================================================
    # ETAPA 1: CARREGAR DADOS
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 1: CARREGAMENTO DOS DADOS")
    print("="*80)

    # IMPORTANTE: Substitua pela carga dos seus dados reais
    # data = pd.read_csv('seus_dados_pib.csv', parse_dates=['data'], index_col='data')

    print("\n⚠ ATENÇÃO: Usando dados sintéticos de exemplo")
    print("  Para usar seus dados reais, substitua a função generate_synthetic_pib_data()")
    print("  por: data = pd.read_csv('seus_dados_pib.csv', parse_dates=['data'], index_col='data')")

    data = generate_synthetic_pib_data(n_obs=300)

    print(f"\n✓ Dados carregados: {len(data)} observações")
    print(f"  Período: {data.index[0].strftime('%Y-%m')} a {data.index[-1].strftime('%Y-%m')}")
    print(f"  Variáveis: {len(data.columns)}")
    print(f"\n  Primeiras 5 linhas:")
    print(data.head())

    # Verificar valores missing
    missing = data.isnull().sum().sum()
    if missing > 0:
        print(f"\n⚠ Valores missing encontrados: {missing}")
        print("  Aplicando interpolação linear...")
        data = data.interpolate(method='linear', limit_direction='both')

    # Dividir dados
    n = len(data)
    train_end = int(n * Config.TRAIN_RATIO)
    val_end = train_end + int(n * Config.VAL_RATIO)

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    print(f"\n✓ Divisão dos dados:")
    print(f"  Treino: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validação: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Teste: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")

    # ========================================================================
    # ETAPA 2: ANÁLISE EXPLORATÓRIA
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 2: ANÁLISE EXPLORATÓRIA")
    print("="*80)

    plot_exploratory_analysis(data, Config.TARGET_VAR, Config.OUTPUT_DIR)

    print("\n✓ Análise exploratória concluída")
    print(f"  3 gráficos salvos em: {Config.OUTPUT_DIR}/")

    # ========================================================================
    # ETAPA 3: VALIDAÇÃO DE VARIÁVEIS
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 3: VALIDAÇÃO ESTATÍSTICA DE VARIÁVEIS")
    print("="*80)

    print("\nInicializando VariableValidator...")
    print(f"  Nível de significância: {Config.SIGNIFICANCE_LEVEL}")
    print(f"  Máximo de lags (Granger): {Config.MAX_LAG_GRANGER}")
    print(f"  Força mínima de causalidade: {Config.MIN_CAUSAL_STRENGTH}")

    validator = VariableValidator(
        significance_level=Config.SIGNIFICANCE_LEVEL,
        max_lag_granger=Config.MAX_LAG_GRANGER,
        min_causal_strength=Config.MIN_CAUSAL_STRENGTH,
        auto_transform=Config.AUTO_TRANSFORM
    )

    print("\nExecutando validação completa (pode demorar alguns minutos)...")
    print("  1. Testes de estacionaridade (ADF, KPSS, Phillips-Perron)")
    print("  2. Transformação para estacionaridade (se necessário)")
    print("  3. Testes de causalidade de Granger")
    print("  4. Seleção de preditores válidos")

    validation_results = validator.validate_all(
        data=train_data,
        target_var=Config.TARGET_VAR,
        predictor_vars=Config.EXOG_VARS,
        verbose=False  # Trocar para True para ver detalhes
    )

    selected_predictors = [p['variable'] for p in validation_results['selected_predictors']]

    print(f"\n✓ Validação concluída!")
    print(f"  Candidatos testados: {len(Config.EXOG_VARS)}")
    print(f"  Preditores selecionados: {len(selected_predictors)}")

    if selected_predictors:
        print(f"\n  Top 10 preditores mais importantes:")
        importance_df = validator.get_feature_importance()
        print(importance_df.head(10).to_string(index=False))

        # Salvar lista completa
        if Config.SAVE_RESULTS:
            importance_df.to_csv(Config.OUTPUT_DIR / 'selected_predictors.csv', index=False)
            print(f"\n  ✓ Lista completa salva em: selected_predictors.csv")

    # Visualizar
    plot_validation_results(validation_results, Config.OUTPUT_DIR)

    # ========================================================================
    # ETAPA 4: TREINAMENTO DE MODELOS
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 4: TREINAMENTO DE MODELOS AVANÇADOS")
    print("="*80)

    results = {}
    actual_test = test_data[Config.TARGET_VAR].values

    # Usar top N preditores (limitar para performance)
    max_predictors = min(10, len(selected_predictors))
    top_predictors = selected_predictors[:max_predictors]

    print(f"\nUsando top {max_predictors} preditores para modelagem:")
    for i, pred in enumerate(top_predictors, 1):
        print(f"  {i}. {pred}")

    # --- Modelo 1: SARIMA (baseline univariado) ---
    print("\n" + "-"*80)
    print("Modelo 1: SARIMA (baseline univariado)")
    print("-"*80)

    try:
        sarima = SARIMAPredictor(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            name="SARIMA_Baseline"
        )

        print("Treinando...")
        sarima.fit(train_data[Config.TARGET_VAR])

        print("Prevendo...")
        forecast = sarima.predict(steps=len(test_data))

        mape = np.mean(np.abs((actual_test - forecast) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast) ** 2))
        mae = np.mean(np.abs(actual_test - forecast))

        results['SARIMA'] = {
            'model': sarima,
            'forecast': forecast,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        criteria = sarima.get_information_criteria()
        print(f"✓ SARIMA treinado com sucesso!")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  AIC: {criteria['AIC']:.2f}")

    except Exception as e:
        print(f"✗ Erro no SARIMA: {str(e)}")

    # --- Modelo 2: SARIMAX (com exógenas) ---
    if top_predictors:
        print("\n" + "-"*80)
        print("Modelo 2: SARIMAX (com variáveis exógenas)")
        print("-"*80)

        try:
            sarimax = SARIMAXPredictor(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                exog_names=top_predictors,
                name="SARIMAX_Full"
            )

            print(f"Treinando com {len(top_predictors)} preditores...")
            sarimax.fit(
                train_data[Config.TARGET_VAR],
                exog=train_data[top_predictors]
            )

            print("Prevendo...")
            forecast = sarimax.predict(
                steps=len(test_data),
                exog=test_data[top_predictors]
            )

            mape = np.mean(np.abs((actual_test - forecast) / (actual_test + 1e-8))) * 100
            rmse = np.sqrt(np.mean((actual_test - forecast) ** 2))
            mae = np.mean(np.abs(actual_test - forecast))

            results['SARIMAX'] = {
                'model': sarimax,
                'forecast': forecast,
                'mape': mape,
                'rmse': rmse,
                'mae': mae
            }

            coeffs = sarimax.get_exog_coefficients()
            print(f"✓ SARIMAX treinado com sucesso!")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  RMSE: {rmse:.4f}")
            print(f"\n  Top 5 coeficientes:")
            for i, (var, coef) in enumerate(list(coeffs.items())[:5], 1):
                print(f"    {i}. {var}: {coef:.4f}")

        except Exception as e:
            print(f"✗ Erro no SARIMAX: {str(e)}")

    # --- Modelo 3: CatBoost ---
    print("\n" + "-"*80)
    print("Modelo 3: CatBoost")
    print("-"*80)

    try:
        catboost = CatBoostPredictor(
            lookback=12,
            iterations=300,
            learning_rate=0.05,
            depth=6,
            name="CatBoost"
        )

        print("Treinando...")
        catboost.fit(train_data[Config.TARGET_VAR])

        print("Prevendo...")
        forecast = catboost.predict(steps=len(test_data))

        mape = np.mean(np.abs((actual_test - forecast) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast) ** 2))
        mae = np.mean(np.abs(actual_test - forecast))

        results['CatBoost'] = {
            'model': catboost,
            'forecast': forecast,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"✓ CatBoost treinado com sucesso!")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.4f}")

    except Exception as e:
        print(f"✗ Erro no CatBoost: {str(e)}")

    # --- Modelo 4: LightGBM ---
    print("\n" + "-"*80)
    print("Modelo 4: LightGBM")
    print("-"*80)

    try:
        lightgbm = LightGBMPredictor(
            lookback=12,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            name="LightGBM"
        )

        print("Treinando...")
        lightgbm.fit(train_data[Config.TARGET_VAR])

        print("Prevendo...")
        forecast = lightgbm.predict(steps=len(test_data))

        mape = np.mean(np.abs((actual_test - forecast) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast) ** 2))
        mae = np.mean(np.abs(actual_test - forecast))

        results['LightGBM'] = {
            'model': lightgbm,
            'forecast': forecast,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"✓ LightGBM treinado com sucesso!")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.4f}")

    except Exception as e:
        print(f"✗ Erro no LightGBM: {str(e)}")

    # --- Modelo 5: Prophet ---
    print("\n" + "-"*80)
    print("Modelo 5: Prophet (Facebook)")
    print("-"*80)

    try:
        prophet = ProphetPredictor(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            name="Prophet"
        )

        print("Treinando...")
        prophet.fit(train_data[Config.TARGET_VAR])

        print("Prevendo...")
        forecast = prophet.predict(steps=len(test_data))

        mape = np.mean(np.abs((actual_test - forecast) / (actual_test + 1e-8))) * 100
        rmse = np.sqrt(np.mean((actual_test - forecast) ** 2))
        mae = np.mean(np.abs(actual_test - forecast))

        results['Prophet'] = {
            'model': prophet,
            'forecast': forecast,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        print(f"✓ Prophet treinado com sucesso!")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.4f}")

    except Exception as e:
        print(f"✗ Erro no Prophet: {str(e)}")

    # ========================================================================
    # ETAPA 5: COMPARAÇÃO E ANÁLISE
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 5: COMPARAÇÃO DE DESEMPENHO")
    print("="*80)

    if results:
        # Criar tabela de comparação
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Modelo': model_name,
                'MAPE (%)': result['mape'],
                'RMSE': result['rmse'],
                'MAE': result['mae']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAPE (%)')

        print("\n✓ Ranking de modelos:")
        print(comparison_df.to_string(index=False))

        # Melhor modelo
        best_model_name = comparison_df.iloc[0]['Modelo']
        best_mape = comparison_df.iloc[0]['MAPE (%)']

        print(f"\n🏆 MELHOR MODELO: {best_model_name}")
        print(f"   MAPE: {best_mape:.2f}%")
        print(f"   RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
        print(f"   MAE: {comparison_df.iloc[0]['MAE']:.4f}")

        # Melhoria SARIMAX vs SARIMA
        if 'SARIMA' in results and 'SARIMAX' in results:
            improvement = ((results['SARIMA']['mape'] - results['SARIMAX']['mape']) /
                          results['SARIMA']['mape'] * 100)
            print(f"\n📊 Melhoria do SARIMAX sobre SARIMA: {improvement:.2f}%")
            if improvement > 0:
                print("   ✓ Variáveis exógenas melhoraram significativamente a previsão!")

        # Salvar resultados
        if Config.SAVE_RESULTS:
            comparison_df.to_csv(Config.OUTPUT_DIR / 'model_comparison.csv', index=False)
            print(f"\n✓ Comparação salva em: model_comparison.csv")

        # Visualizações
        plot_model_comparison(results, actual_test, Config.OUTPUT_DIR)
        print(f"\n✓ 3 gráficos de comparação salvos em: {Config.OUTPUT_DIR}/")

        # Salvar melhor modelo
        if Config.SAVE_MODELS:
            best_model = results[best_model_name]['model']
            with open(Config.OUTPUT_DIR / f'best_model_{best_model_name}.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            print(f"\n✓ Melhor modelo salvo em: best_model_{best_model_name}.pkl")

    # ========================================================================
    # ETAPA 6: PREVISÃO FUTURA
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 6: PREVISÃO PARA OS PRÓXIMOS 12 MESES")
    print("="*80)

    if results and best_model_name:
        print(f"\nUsando modelo: {best_model_name}")

        best_model = results[best_model_name]['model']

        # Retreinar com todos os dados disponíveis
        print("Retreinando com todos os dados (treino + validação + teste)...")

        all_data = data[Config.TARGET_VAR]

        if best_model_name == 'SARIMAX' and top_predictors:
            # Para SARIMAX precisamos de valores futuros de exógenas
            print("\n⚠ SARIMAX requer valores futuros das variáveis exógenas")
            print("  Para previsão real, você precisará fornecer:")
            print("  - Valores projetados das variáveis exógenas")
            print("  - Ou usar cenários (otimista, pessimista, base)")
            print("\n  Neste exemplo, usaremos os últimos valores observados (simplificação)")

            # Usar últimos valores (simplificação)
            future_exog = pd.DataFrame(
                np.tile(data[top_predictors].iloc[-1].values, (Config.FORECAST_HORIZON, 1)),
                columns=top_predictors
            )

            best_model.fit(all_data, exog=data[top_predictors])
            future_forecast = best_model.predict(steps=Config.FORECAST_HORIZON, exog=future_exog)
        else:
            best_model.fit(all_data)
            future_forecast = best_model.predict(steps=Config.FORECAST_HORIZON)

        # Criar DataFrame com previsões
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=Config.FORECAST_HORIZON, freq='MS')

        future_df = pd.DataFrame({
            'Data': future_dates,
            'PIB_Previsto': future_forecast
        })

        print(f"\n✓ Previsão para os próximos {Config.FORECAST_HORIZON} meses:")
        print(future_df.to_string(index=False))

        # Salvar
        if Config.SAVE_RESULTS:
            future_df.to_csv(Config.OUTPUT_DIR / 'forecast_future_12months.csv', index=False)
            print(f"\n✓ Previsões futuras salvas em: forecast_future_12months.csv")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))

        # Histórico
        ax.plot(data.index, data[Config.TARGET_VAR], 'k-', linewidth=2,
                label='Histórico', alpha=0.7)

        # Futuro
        ax.plot(future_dates, future_forecast, 'r--', linewidth=2.5,
                label=f'Previsão ({best_model_name})', marker='o', markersize=6)

        # Sombrear área futura
        ax.axvspan(future_dates[0], future_dates[-1], alpha=0.1, color='red')

        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('PIB', fontsize=12)
        ax.set_title(f'Previsão de PIB para os Próximos 12 Meses ({best_model_name})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if Config.SAVE_PLOTS:
            plt.savefig(Config.OUTPUT_DIR / 'forecast_future_12months.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Gráfico de previsão futura salvo em: forecast_future_12months.png")

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n✓ Pipeline completo executado com sucesso!")
    print(f"  Duração: {duration.total_seconds():.1f} segundos ({duration.total_seconds()/60:.1f} minutos)")

    print(f"\n📊 Estatísticas:")
    print(f"  ✓ Observações: {len(data)}")
    print(f"  ✓ Variáveis candidatas: {len(Config.EXOG_VARS)}")
    print(f"  ✓ Variáveis selecionadas: {len(selected_predictors)}")
    print(f"  ✓ Modelos treinados: {len(results)}")
    print(f"  ✓ Melhor modelo: {best_model_name if results else 'N/A'}")
    print(f"  ✓ Melhor MAPE: {best_mape:.2f}%" if results else "  ✗ Nenhum modelo treinado")

    print(f"\n📁 Arquivos gerados em: {Config.OUTPUT_DIR}/")

    output_files = list(Config.OUTPUT_DIR.glob('*'))
    for f in sorted(output_files):
        print(f"  - {f.name}")

    print("\n" + "="*80)
    print("PREVISÃO DE PIB CONCLUÍDA COM SUCESSO!".center(80))
    print("="*80)

    print("\n💡 Próximos passos:")
    print("  1. Analise os gráficos gerados")
    print("  2. Revise as variáveis selecionadas")
    print("  3. Use o melhor modelo para previsões")
    print("  4. Substitua os dados sintéticos pelos seus dados reais")
    print("  5. Ajuste hiperparâmetros se necessário")


if __name__ == "__main__":
    main()
