# Conhecendo os Dados e Preparação para Modelagem

Nesta seção, realizamos uma Análise Exploratória de Dados (EDA) aprofundada e a Engenharia de Features para compreender o comportamento dos pacientes do sistema de saúde (SUS) de Vitória/ES. O objetivo final desta etapa é extrair *insights* sobre o absenteísmo (variável alvo `No-show`) e preparar um *dataset* íntegro, ético e matematicamente formatado para a etapa de Machine Learning.

## 1. Limpeza e Tratamento Inicial

O dataset original mostrou-se bastante íntegro, sem valores nulos ou linhas duplicadas irrecuperáveis. No entanto, na etapa de detecção de anomalias, identificamos um erro de digitação no sistema médico: um paciente com idade `-1`. Esse registro único foi removido (passando de 110.527 para 110.526 linhas) para não comprometer as medidas de dispersão. Também realizamos a Engenharia de Variáveis, transformando as datas brutas de agendamento e consulta na variável contínua `DiasEspera`.

## 2. Medidas de Tendência Central e Dispersão

Avaliamos as variáveis quantitativas do dataset e seus *outliers*:

* **Idade (Age):** A média de idade dos pacientes é de 37 anos, com a mediana também em 37. A dispersão vai de 0 (recém-nascidos, que representam um pico de comparecimento) até *outliers* naturais de longevidade (115 anos), que foram mantidos por serem dados reais do sistema de saúde.
* **Dias de Espera (DiasEspera):** Identificamos que a mediana de espera para quem comparece é de apenas 2 dias (com média de 8,8 dias). Para os faltosos, a média salta para 15,8 dias, com o Intervalo Interquartil (IQR) muito mais alongado, evidenciando o peso do tempo no abandono da consulta.

## 3. Visualizações e Trechos de Código

Utilizamos histogramas para entender a distribuição etária, gráficos de barras para conversão de categorias e *boxplots* para avaliar a dispersão cruzada. Abaixo, destacamos o trecho de código responsável por criar a variável de tempo e gerar a análise visual de dispersão que baseou nossa principal descoberta:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Feature Engineering: Criando a variável de tempo de espera
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df["DiasEspera"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

# Removendo inconsistências temporais (consultas agendadas para o passado)
df = df[df["DiasEspera"] >= 0]

# 2. Visualização de Dispersão e Outliers (Box plot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='No-show', y='DiasEspera', palette=['#4C72B0', '#DD8452'])
plt.title('Dias de Espera vs. Status de Comparecimento')
plt.xlabel('Faltou? (0 = Não, 1 = Sim)')
plt.ylabel('Dias de Espera')
plt.show()
```
*(Nota: O código fonte completo, contendo matrizes de correlação de Spearman, histogramas e tratamentos, encontra-se disponível no link: https://colab.research.google.com/drive/1GR-eJyyqYYrNghVvF3EF_UwG9ITsuQOt?usp=sharing#scrollTo=HsgyBNiINrEG).*

---

## Descrição dos Achados

A análise exploratória revelou que o absenteísmo médico possivelmente não é um evento aleatório, mas um comportamento influenciado por fatores demográficos, temporais e geográficos:

1. **Desbalanceamento da Variável Alvo:** A base de dados apresenta um desbalanceamento natural claro, com 79,8% de comparecimento e 20,2% de faltas. Este insight é crucial, pois exigirá técnicas de rebalanceamento (como SMOTE ou Class Weights) e a priorização de métricas como F1-Score e Recall na etapa de Machine Learning, em detrimento da simples Acurácia.
2. **Indício de Associação Temporal (A Regra do Tempo):** A análise validou uma associação forte (Spearman: 0.28) entre o tempo de espera e as faltas. Consultas marcadas para o mesmo dia ou para a mesma semana possuem uma taxa de adesão alta. Agendamentos longos são os maiores propulsores de evasão.
3. **Perfil Etário:** Cruzando a idade com as faltas, notamos que o absenteísmo se concentra no público jovem/adulto (a mediana de idade de quem falta é de 33 anos). Bebês (levados pelos pais) e idosos possuem rotinas mais flexíveis e apresentam alta taxa de comparecimento.
4. **O Paradoxo das Comorbidades:** Verificou-se uma relação moderada e contraintuitiva à primeira vista: pacientes sinalizados com doenças crônicas faltam menos. Enquanto a taxa geral de faltas é de 20,2%, entre os hipertensos ela cai para 17,3%, evidenciando o engajamento gerado pela dependência de uma rotina médica contínua.
5. **Correção do Viés Geográfico:** Inicialmente, bairros populosos como Jardim Camburi lideravam em faltas absolutas. No entanto, ao calcular a Taxa Proporcional de Absenteísmo (Faltas / Total de Consultas do Bairro), descobrimos que regiões como Santos Dumont, Santa Clara e Itararé são hotspots de risco proporcional, indicando possíveis problemas estruturais (como fricção de mobilidade ou inflexibilidade de trabalho dos moradores locais).

---

## Preparação Final para Modelagem (Dataset de ML)

Para a transição da Etapa 2 para a Etapa 3 (Modelagem), o dataset precisou ser codificado.

### Dicionário de Variáveis e Transformações

| Variável | Papel no ML | Transformação Aplicada | Justificativa Técnica |
| :--- | :--- | :--- | :--- |
| **NoShow_numeric** | Alvo (Target) | Label Encoding (0/1) | A variável que o modelo aprenderá a prever. |
| **DiasEspera** | Feature | Engenharia de Datas | O principal motor preditivo, extraído das datas brutas. |
| **Age** | Feature | Limpeza de Ruído | Escala numérica mantida. Correlação negativa com o alvo. |
| **Gender_numeric** | Feature | Label Encoding | Transformação de 'M/F' para binário (0/1). |
| **Neighbourhood_Risk** | Feature | Target Encoding | Substituição do nome do bairro pela sua taxa histórica de evasão, capturando o risco sem explodir a dimensionalidade. |
| **SMS_received, Hipertension, Diabetes, etc** | Features | Nenhuma (Original) | Variáveis binárias mantidas para avaliar o contexto social e clínico. |

### Mitigação de Vazamento de Dados (Data Leakage)

O vazamento de dados ocorre quando o modelo recebe informações que só estariam disponíveis no futuro. Para evitar isso e garantir a generalização do algoritmo, as seguintes colunas foram excluídas do dataset final:

* **PatientId e AppointmentID:** Removidas. São identificadores únicos. Mantê-las faria o modelo decorar o paciente (overfitting) em vez de aprender padrões estatísticos.
* **ScheduledDay e AppointmentDay:** Removidas em sua forma bruta após a extração de DiasEspera, pois modelos não calculam datas diretamente.
* **Neighbourhood (Texto):** Removida após a aplicação do Target Encoding (conversão para risco percentual).

---

## Ética, Privacidade (LGPD) e Impacto Social

Trabalhar com dados do Sistema Único de Saúde (SUS) exige um protocolo rigoroso de governança. O dataset processado envolve características sensíveis como condições de saúde (comorbidades, deficiências) e marcadores socioeconômicos (Bolsa Família).

Estabelecemos as seguintes salvaguardas para o uso deste modelo de IA:

1. **Anonimização (LGPD):** Com o descarte definitivo do PatientId e do AppointmentID, o dataset de modelagem foi anonimizado. É impossível realizar a reidentificação dos indivíduos, garantindo conformidade com a minimização de dados exigida pela LGPD.
2. **Mitigação de Vieses Discriminatórios:** Marcadores como Scholarship (proxy de baixa renda) e Neighbourhood (proxy de região) não devem ser usados pelo algoritmo para punir populações vulneráveis. Monitoraremos métricas de Fairness (Justiça Algorítmica) na Etapa 3 para garantir que o modelo não assuma um comportamento discriminatório contra moradores de áreas periféricas ou jovens adultos.
3. **Riscos do Overbooking Responsável:** A IA que será construída apontará quais agendamentos têm alta probabilidade de falta, permitindo a abertura de vagas de encaixe (overbooking). Contudo, essa predição deve atuar como suporte à decisão, não como um sistema automatizado cego (necessitando da abordagem Human-in-the-Loop). O excesso de confiança no modelo sem avaliar o contexto local pode gerar superlotação nos postos de saúde, prejudicando a qualidade do atendimento e penalizando pacientes que sofreram imprevistos justificados (como falhas no transporte público).

---

## Ferramentas Utilizadas

* **Python (v3.x):** Linguagem padrão para manipulação estatística e modelagem.
* **Google Colab:** Ambiente interativo que permitiu a mescla de código em tempo real e documentação em Markdown.
* **Pandas & NumPy:** Essenciais para a manipulação tabular, engenharia de variáveis (datas) e agregações descritivas.
* **Matplotlib & Seaborn:** Empregadas na geração visual (histogramas, heatmaps e boxplots), permitindo validar graficamente as hipóteses estatísticas.
* **Git & GitHub:** Versionamento, histórico de alterações e documentação do projeto.
