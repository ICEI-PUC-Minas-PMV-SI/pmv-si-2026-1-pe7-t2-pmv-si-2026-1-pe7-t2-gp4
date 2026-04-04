# Conhecendo os dados

Nesta seção, realizamos uma Análise Exploratória de Dados (EDA) aprofundada para compreender o comportamento dos pacientes do sistema de saúde (SUS) de Vitória/ES, focando em identificar os fatores que levam ao absenteísmo (variável alvo `No-show`).

**1. Limpeza e Tratamento Inicial**
O dataset original mostrou-se bastante íntegro, sem valores nulos ou linhas duplicadas. No entanto, na etapa de detecção de anomalias e _outliers_, identificamos um erro de digitação no sistema médico: um paciente com idade `-1`. Esse registro foi removido para não comprometer as medidas de dispersão. Também realizamos a Engenharia de Variáveis, transformando as datas de agendamento e consulta na variável contínua `DiasEspera`.

**2. Medidas de Tendência Central e Dispersão**
Avaliamos as variáveis quantitativas do dataset:
* **Idade (Age):** A média de idade dos pacientes é de 37 anos, com a mediana também em 37. A dispersão vai de 0 (recém-nascidos, que representam um pico no histograma) até _outliers_ naturais de longevidade (115 anos).
* **Dias de Espera (DiasEspera):** Identificamos que a mediana de espera para quem comparece é muito próxima a 0 (consultas no mesmo dia), com média de 14,0 dias. Para os faltosos, a média sobe para 16,2 dias, com o Intervalo Interquartil (IQR) muito mais alongado.

**3. Visualizações e Trechos de Código**
Utilizamos histogramas para entender a distribuição etária, gráficos de barras para contagem de categorias (bairros e comorbidades) e box plots para avaliar a dispersão cruzada com a variável alvo.

Abaixo, destacamos o trecho de código responsável por criar a variável de tempo e gerar a análise visual de dispersão (Box plot) que baseou nossa principal descoberta:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Feature Engineering: Criando a variável de tempo de espera
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df["DiasEspera"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

# Removendo inconsistências (consultas agendadas para o passado)
df = df[df["DiasEspera"] >= 0]

# 2. Visualização de Dispersão e Outliers (Box plot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='No-show', y='DiasEspera', palette=['#4C72B0', '#DD8452'])
plt.title('Dias de Espera vs. Status de Comparecimento')
plt.xlabel('Faltou? (0 = Não, 1 = Sim)')
plt.ylabel('Dias de Espera')
plt.show()
```
*(Nota: O código fonte completo, contendo matrizes de correlação, histogramas, estatísticas descritivas e tratamentos, encontra-se disponível no link: https://colab.research.google.com/drive/1GR-eJyyqYYrNghVvF3EF_UwG9ITsuQOt?usp=sharing.*

---

## Descrição dos achados

A análise exploratória revelou que o absenteísmo médico não é um evento puramente aleatório, mas sim um comportamento influenciado por fatores demográficos, de saúde e sistêmicos. Os principais achados incluem:

1. **Desbalanceamento da Variável Alvo:** A base de dados apresenta um desbalanceamento claro, com **79,8% de comparecimento** e **20,2% de faltas**. Este é um insight crucial, pois exigirá técnicas de rebalanceamento (como SMOTE ou _Class Weights_) e o uso de métricas como F1-Score na etapa de Machine Learning.
2. **Correlação Temporal (A Regra do Tempo):** Existe uma correlação positiva forte entre o tempo de espera e o absenteísmo. Consultas marcadas para o mesmo dia ou semana possuem adesão quase total. O risco de falta cresce exponencialmente em agendas com mais de 16 dias de antecedência.
3. **Perfil Etário:** Cruzando a idade com as faltas, notamos que o absenteísmo se apresenta, principalmente, no público jovem adulto. A mediana de idade de quem falta é de 33 anos, enquanto quem comparece tem mediana de 40 anos. Bebês (levados pelos pais) e idosos possuem alta taxa de comparecimento.
4. **O Paradoxo das Comorbidades:** Verificou-se uma relação moderada e contraintuitiva à primeira vista: pacientes sinalizados com doenças crônicas faltam menos. Enquanto a taxa geral de faltas é de 20,2%, entre os hipertensos ela cai para 17,3%, evidenciando o engajamento gerado pela necessidade de rotina médica rígida.
5. **Concentração Geográfica:** Bairros com alta densidade populacional (como Jardim Camburi e Maria Ortiz) lideram o ranking absoluto de faltas. Isso indica que campanhas de redução de absenteísmo (como disparos de SMS automáticos) devem ser priorizadas para pacientes dessas regiões, especialmente se cruzadas com agendas longas.

---

## Ferramentas utilizadas

Para garantir a confiabilidade, reprodutibilidade e eficiência nas análises descritivas e exploratórias, o projeto foi desenvolvido utilizando o seguinte ecossistema tecnológico focado em Ciência de Dados:

* **Linguagem de Programação:** `Python (v3.x)` - Escolhida por ser o padrão na indústria, possuindo suporte robusto para manipulação estatística e modelagem preditiva.
* **Ambiente de Desenvolvimento:** `Google Colab` - Utilizado para a criação de um fluxo de trabalho iterativo, permitindo a mescla de execução de blocos de código com documentação rica em Markdown.
* **Pandas:** Biblioteca principal utilizada para a manipulação tabular dos dados, limpeza (identificação de nulos e duplicados), feature engineering (cálculo de datas de espera) e agregações estatísticas (`.describe()`, `.value_counts()`).
* **NumPy:** Utilizada como suporte para operações matemáticas de baixo nível e vetorização de matrizes durante a avaliação quantitativa dos dados.
* **Matplotlib & Seaborn:** Bibliotecas empregadas de forma conjunta para a visualização de dados. O Seaborn foi essencial para a geração de gráficos estatísticos avançados (como Box plots com separação por classes e mapas de calor para correlações), enquanto o Matplotlib garantiu a customização de eixos, paletas de cores e dimensionamento (`figsize`).
* **Git & GitHub:** Ferramentas utilizadas para o controle de versão, documentação de progresso (README) e armazenamento seguro do código fonte.

