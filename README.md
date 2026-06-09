<p align="center">
  <img src="reorbita-logo.png" alt="REORBITA Logo" width="160"/>
</p>

<h1 align="center">REORBITA — Inteligência Orbital com Machine Learning</h1>

<p align="center">
  <em>"Construímos um cemitério a 400 km de altura. A REORBITA transforma esse cemitério em uma oficina."</em>
</p>

<p align="center">
  <strong>Global Solution 2026.1 · FIAP — Faculdade de Informática e Administração Paulista</strong><br>
  Disciplina: Inteligência Artificial e Machine Learning · Turma: 3ESS
</p>

---

---

## Grupo

| Nome | RM |
|---|---|
| Bento Rangel | RM559124 |
| Eric Yuji | RM554869 |
| Higor Batista | RM558907 |
| Kaue Pires | RM554403 |
| Ricardo Di Tilia | RM555155 |

---

## Sobre o Projeto

Mais de **60% dos satélites** lançados na última década virarão lixo espacial prematuramente — por baterias degradadas, falhas pontuais ou eventos solares extremos. Cada satélite perdido representa entre **USD 80M e USD 400M** descartados sem possibilidade de reparo.

A **REORBITA** é um ecossistema integrado de manutenção orbital sustentado por três pilares:

| Pilar | Descrição |
|---|---|
| 🧠 **Plataforma de Inteligência Orbital** | Digital Twin de cada satélite com IA preditiva — *este repositório* |
| 🤖 **Frota de Robôs Modulares** | Veículos orbitais especializados para reparo, reabastecimento e remoção de detritos |
| 🔌 **Protocolo Orbit-Ready** | Padrão aberto de design — o "USB-C dos satélites" — com módulos substituíveis e interfaces universais |

Este repositório implementa o **motor de Machine Learning** que alimenta o primeiro pilar: previsão de risco de flares solares com 24 horas de antecedência e cálculo de risco sistêmico individual por satélite.

---

## O Problema de ML

Os flares solares do tipo X — os mais destrutivos para satélites — ocorrem em **menos de 2% dos dias** no histórico. O período de teste (2023–2024) coincide com o pico do Ciclo Solar 25, o mais intenso das últimas décadas, com frequência **3x acima da média histórica**. Esse fenômeno, chamado de **concept drift**, é o principal desafio técnico do projeto.

---

## Resultados Principais

| Métrica | Valor |
|---|---|
| **F1 Weighted** (teste 2023–2024) | **0.753** |
| **TS-CV F1** (validação temporal) | **0.833** |
| **Recall flares X** | **44%** — 8 de 18 eventos detectados com 24h de antecedência |
| **Modelo final** | Regressão Logística · threshold = 0.46 |
| **Frota simulada** | 50 satélites · USD ~3.634M em ativos protegidos |

### Por que Recall é a métrica que importa?

| Tipo de erro | Consequência | Custo estimado |
|---|---|---|
| Falso alarme (FP) | Satélite entra em stand-by preventivo | ~USD 1M |
| Falha não detectada (FN) | Satélite perdido permanentemente | ~USD 200–400M |

O threshold foi calibrado via **curva Precision-Recall** para priorizar a detecção de eventos reais, aceitando falsos alarmes como custo aceitável diante do risco evitado.

---

## Estrutura do Repositório

```
reorbita-gs-iaml/
│
├── README.md                          ← este arquivo
├── requirements.txt                   ← dependências Python
├── REORBITA_IA_ML.ipynb              ← notebook principal (executável)
│
├── data/
│   ├── daily_solar_data.csv           ← dataset original (NOAA via Kaggle)
│   └── dataset_telemetria_satelites.csv  ← gerado ao rodar o notebook (Digital Twin)
│
└── models/
    └── reorbita_solar_model.pkl       ← modelo exportado via joblib
```

---

## Como Rodar

### Pré-requisitos

- Python 3.8 ou superior (desenvolvido em Python 3.14.2)
- pip

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/reorbita-gs-iaml.git
cd reorbita-gs-iaml

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Coloque o dataset na pasta correta
# Baixe daily_solar_data.csv do Kaggle (link abaixo) e mova para /data
```

### Dataset

O dataset **Space Weather: Solar + Geomagnetic Indices** está disponível publicamente no Kaggle:

🔗 [https://www.kaggle.com/datasets/erevear/space-weather-solar-geomagnetic-indices](https://www.kaggle.com/datasets/erevear/space-weather-solar-geomagnetic-indices)

Arquivo necessário: `daily_solar_data.csv`

### Executando o Notebook

```bash
jupyter notebook REORBITA_IA_ML.ipynb
```

Execute as células em ordem. O notebook é **autocontido** — todas as etapas estão documentadas e comentadas, do carregamento dos dados até a exportação do modelo.

Ao rodar completamente, dois artefatos são gerados automaticamente:
- `models/reorbita_solar_model.pkl` — modelo treinado exportado via joblib
- `data/dataset_telemetria_satelites.csv` — saída do Digital Twin com status da frota simulada

---

## Pipeline de ML

```
Dados brutos (NOAA)
        │
        ▼
1. Limpeza & Tratamento
   └─ Substituição de -1 por NaN
   └─ Forward fill temporal (persistência solar)
        │
        ▼
2. Feature Engineering (27 features)
   └─ Lags temporais (D-1, D-2, D-3)
   └─ Médias móveis de 7 dias (roll7)
   └─ Deltas (variação dia a dia)
        │
        ▼
3. Variável-Alvo (D+1)
   └─ Classe 0: Baixo   — sem flare M/X relevante
   └─ Classe 1: Moderado — flare M ou C>2
   └─ Classe 2: Alto    — qualquer flare X
        │
        ▼
4. Divisão Temporal
   └─ Treino: 1997–2022
   └─ Teste:  2023–2024 (período de máximo solar)
        │
        ▼
5. Treinamento (TimeSeriesSplit, n=5 folds)
   └─ SMOTE dentro de cada fold (sem data leakage)
   └─ 4 modelos avaliados:
      · Regressão Logística ✓ (selecionado)
      · Random Forest
      · Árvore de Decisão
      · KNN
        │
        ▼
6. Threshold Tuning
   └─ Curva Precision-Recall
   └─ Threshold ótimo: 0.46
        │
        ▼
7. Digital Twin
   └─ Risco sistêmico individual por satélite
   └─ Frota simulada: 50 satélites
        │
        ▼
Modelo exportado (.pkl) + Alertas por satélite (.csv)
```

---

## Decisões Técnicas Relevantes

### Por que Regressão Logística venceu o Random Forest?

O Random Forest — modelo mais complexo — obteve **F1 = 0.000 na classe Alto** durante os testes: identificou zero flares X corretamente no período de avaliação. O motivo é **concept drift**: o período de teste (2023–2024) coincide com o pico do Ciclo Solar 25, o mais intenso das últimas décadas, com frequência de flares X 3x acima da média histórica. O Random Forest memorizou os padrões dos ciclos anteriores e falhou em generalizar para esse novo regime de atividade solar — comportamento típico de modelos de alta capacidade diante de mudanças de distribuição.

A Regressão Logística, por aprender relações lineares mais simples e generalizáveis, mostrou maior robustez à mudança de distribuição. Combinada ao threshold calibrado em 0.46 via curva Precision-Recall, alcançou **F1 Weighted de 0.753** e **recall de 44%** na classe Alto — detectando 8 dos 18 flares X do período de teste com 24 horas de antecedência. Em produção, o retreino periódico do modelo com dados recentes é o mecanismo previsto para mitigar o concept drift de forma contínua.

### Por que SMOTE apenas no treino?

Aplicar SMOTE antes do split de validação contamina as métricas — amostras sintéticas derivadas do conjunto de validação acabam no treino, inflando artificialmente o desempenho. Aqui, o SMOTE é aplicado via `imblearn.Pipeline` dentro de cada fold do `TimeSeriesSplit`, garantindo que nenhuma amostra sintética influencia a avaliação.

### Por que Forward Fill e não média/mediana?

Condições solares têm alta persistência temporal — manchas solares duram dias ou semanas. O último valor observado é fisicamente mais informativo do que a média histórica para preencher lacunas curtas. O Forward Fill respeita a natureza temporal dos dados.

---

## Digital Twin

O modelo solar é apenas a entrada do sistema. O **Digital Twin** cruza a probabilidade de flare com a telemetria individual de cada satélite:

```python
risco_sistemico = p_flare_x * (1 - saude_bateria) * (1 - eficiencia_painel) * fator_exposicao
```

O mesmo evento solar gera decisões diferentes para cada satélite:

| Satélite | P(Flare X) | Bateria | Painel | Risco Sistêmico |
|---|---|---|---|---|
| SAT-LEGACY-07 (degradado) | 45% | 40% | 70% | 🔴 100% |
| SAT-NEW-23 (novo) | 45% | 95% | 90% | 🟢 22% |

---

## Alinhamento com ODS

| ODS | Contribuição |
|---|---|
| **ODS 9** — Indústria e Infraestrutura | Proteção da infraestrutura orbital crítica |
| **ODS 13** — Ação Climática | Monitoramento de clima espacial; proteção de satélites de previsão do tempo |
| **ODS 11** — Cidades Sustentáveis | Continuidade de GPS, telecomunicações e sistemas de alerta de desastres |

---

## Carregando o Modelo em Produção

```python
import joblib
import pandas as pd

# Carrega o modelo exportado
artefato = joblib.load('models/reorbita_solar_model.pkl')
modelo     = artefato['modelo']
threshold  = artefato['threshold']   # 0.46
features   = artefato['features']    # lista com os 27 nomes das features

# Previsão para novos dados
# X_novo deve conter exatamente as colunas em features, na mesma ordem
probas = modelo.predict_proba(X_novo)
pred   = (probas[:, 2] >= threshold).astype(int)  # 1 = alerta Alto
```

---

## Referências

- **Dataset:** [Space Weather: Solar + Geomagnetic Indices — Kaggle](https://www.kaggle.com/datasets/erevear/space-weather-solar-geomagnetic-indices)
- **NOAA Space Weather Prediction Center:** [https://www.swpc.noaa.gov](https://www.swpc.noaa.gov)
- **NASA Science — Space Weather:** [https://science.nasa.gov/heliophysics/space-weather](https://science.nasa.gov/heliophysics/space-weather)
- **INPE — Instituto Nacional de Pesquisas Espaciais:** [https://www.gov.br/inpe/pt-br](https://www.gov.br/inpe/pt-br)
- **Chawla et al. (2002)** — SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

---

*REORBITA · Global Solution 2026.1 · FIAP*