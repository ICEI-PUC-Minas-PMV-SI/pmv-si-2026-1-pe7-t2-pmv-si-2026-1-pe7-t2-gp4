# Preparação dos dados

Nesta etapa, aplicamos técnicas de pré-processamento para adequar os dados coletados na etapa de Análise Exploratória (EDA) aos requisitos dos algoritmos de *Machine Learning*. As principais transformações realizadas no nosso contexto foram:

* **Limpeza de Dados:** Antes de qualquer transformação, removemos registros logicamente impossíveis, frutos de erro de sistema ou de cadastro. As regras aplicadas e o impacto de cada uma foram:

  | Ordem | Regra | Justificativa | Removidos | Tamanho após |
  |---|---|---|---:|---:|
  | Base bruta | — | *Dataset* original | — | 110.527 |
  | 1 | `DiasEspera < 0` | Consulta antes do agendamento — impossível; erro de sistema | 5 | 110.522 |
  | 2 | `Age < 0` (idade −1) | Erro de digitação no cadastro | 1 | 110.521 |

  O impacto total foi de apenas 6 registros (≈ 0,005% da base). *Outliers* **válidos** (idades acima de 95 anos, esperas superiores a 100 dias) foram **mantidos** por decisão metodológica, por representarem casos reais e relevantes para o estudo do absenteísmo.
* **_Feature Engineering_:** A partir das datas de agendamento e da consulta, criamos a variável `DiasEspera`, que demonstrou ser a característica de maior impacto preditivo para o absenteísmo neste conjunto de dados. Descartamos colunas de identificação que não agregavam valor ao modelo.
* **Transformação de Dados:** Variáveis categóricas, como Gênero e o Status de Comparecimento (alvo), foram codificadas para formatos numéricos (`Gender_numeric` e `NoShow_numeric`), permitindo a interpretação matemática pelos modelos. A variável-alvo foi padronizada sob um único nome, **`NoShow_numeric`** (sem hífen), em todo o pipeline.
* **Separação de dados:** Como aproximadamente 30% dos pacientes possuem **múltiplas consultas** no *dataset*, um `train_test_split` aleatório espalharia consultas de um mesmo paciente entre treino e teste, gerando **vazamento de dados** (*data leakage*). Por isso, dividimos o *dataset* com **`GroupShuffleSplit` agrupando por `PatientId`**: todas as consultas de um mesmo paciente ficam obrigatoriamente do mesmo lado da divisão. Usamos `test_size=0.30` e `random_state=42` (reprodutibilidade), resultando em **77.565 registros de treino e 32.956 de teste**. Uma verificação automática confirma que nenhum paciente aparece simultaneamente nos dois conjuntos.
* **Tratamento de dados desbalanceados:** A variável alvo possui um desbalanceamento natural (aproximadamente 80% de comparecimentos contra 20% de faltas). Optamos por **não** gerar dados sintéticos; em vez disso, utilizamos o balanceamento de pesos de classe (`class_weight`) diretamente no algoritmo, penalizando mais os erros sobre a classe minoritária (a falta). A validação cruzada usada na busca de hiperparâmetros também respeita o agrupamento por paciente, via `GroupKFold` de 5 *folds*.
* **Padronização (*Scaling*):** Para evitar que variáveis com valores absolutos maiores (como `DiasEspera` ou `Age`) tivessem uma influência desproporcional nos modelos sensíveis à escala (como a Regressão Logística do *baseline*), aplicamos o `StandardScaler`. O escalonador foi ajustado exclusivamente nos dados de treino e apenas aplicado aos dados de teste, mitigando o risco de vazamento de dados (*Data Leakage*).

# Descrição do modelo

Após um teste inicial comparando Árvore de Decisão, Regressão Logística e Random Forest, optamos por selecionar e aprofundar a construção do modelo proposto utilizando o algoritmo **Random Forest Classifier** (Floresta Aleatória).

**Princípios de funcionamento e conceitos fundamentais:**
O Random Forest é um algoritmo de aprendizado supervisionado baseado na técnica de *Ensemble* (conjunto). Ele cria múltiplas Árvores de Decisão independentes durante o treinamento, cada uma avaliando subconjuntos aleatórios dos dados e das características. A previsão final é feita por meio de uma "votação" majoritária entre todas as árvores.

**Vantagens e justificativa da escolha:**
O algoritmo foi selecionado por sua robustez contra *overfitting* (memorização dos dados) e por oferecer explicabilidade técnica através da extração de Importância das Variáveis (*Feature Importance*). Essa métrica indicou que, para este modelo e *dataset*, a variável `DiasEspera` representou 76,2% do poder de decisão, seguida por `Age` (13,8%). Variáveis relacionadas a condições preexistentes apresentaram baixa importância preditiva nesta modelagem específica.

**Ajuste dos parâmetros livres:**
Para explorar o espaço de otimização, utilizamos a Validação Cruzada agrupada (`GridSearchCV` com `GroupKFold` de 5 *folds* e `scoring='f1'`). Os melhores hiperparâmetros encontrados foram: profundidade máxima `max_depth=10`, número de estimadores `n_estimators=100`, `min_samples_split=10` e `min_samples_leaf=4`, com **melhor F1 na validação cruzada ≈ 0,440**.

Além dos hiperparâmetros nativos, implementamos um **Ajuste de Limiar (*Threshold Tuning*)**. Por padrão, o algoritmo classifica uma falta se a probabilidade for maior que 50%. Testamos elevar esse limiar para **70%**, tornando o modelo mais conservador (alerta de falta apenas em casos de maior confiança). É importante deixar claro que **70% não é um ótimo matemático**, e sim um ponto de operação de negócio: como mostra a análise formal de *threshold tuning* da Etapa 4 (ver [construcao-de-modelos.md](construcao-de-modelos.md)), em 0,70 o *recall* cai para cerca de 0,20, ou seja, o modelo deixa de sinalizar a maioria das faltas em troca de uma precisão maior. A calibração final do limiar é, portanto, uma decisão da gestão sobre o *trade-off* entre precisão e *recall*.

# Avaliação dos modelos criados

## Métricas utilizadas

A avaliação priorizou métricas adequadas para bases desbalanceadas:

* **Acurácia:** O modelo ajustado atingiu 79% de acurácia. Contudo, essa métrica foi tratada apenas como referência secundária, visto que um algoritmo que previsse comparecimento para todos os registros já atingiria cerca de 80% neste *dataset*.
* **Precisão (*Precision*):** Mede a proporção de Faltas reais entre os casos que o modelo classificou como "Faltosos". É a métrica utilizada para mensurar o volume de Falsos Positivos.
* **Recall (Sensibilidade):** Avalia a proporção de faltosos reais que o modelo conseguiu identificar corretamente no *dataset*.
* **F1-Score:** A média harmônica entre Precisão e Recall. Foi utilizada como a métrica de ranqueamento (`scoring`) durante a otimização de parâmetros.

## Discussão dos resultados obtidos

A avaliação das métricas evidenciou o impacto direto do *Threshold Tuning* na aplicabilidade do modelo para o cenário de **Overbooking Responsável**.

O teste inicial evidenciou o desafio do desbalanceamento, com algoritmos alcançando F1-Scores na faixa de 0.29 a 0.35. Com o Random Forest ajustado para o limiar de 70%, observamos uma alteração no comportamento da Matriz de Confusão. Dos 32.956 registros validados no teste, o modelo identificou 1.005 Faltas Reais (Verdadeiros Positivos) e classificou incorretamente 1.382 casos (Falsos Positivos). Esses resultados indicam uma priorização estatística da **Precisão** (42%) em detrimento do **Recall** (15%). O modelo deixa de sinalizar a maioria das faltas (Falsos Negativos = 5.689), mas demonstra maior margem de acerto quando a previsão de falta ultrapassa o limiar de 70% estabelecido. 

É importante ressaltar que este limite estatístico não é uma regra rígida, mas sim um parâmetro de negócio flexível que reflete o clássico *trade-off* entre Precisão e Recall. Embora o valor de 70% tenha sido escolhido para este estudo por representar um ponto de equilíbrio viável, limiares mais conservadores (como 80% ou 90%) poderiam ser adotados pela gestão da clínica. Aumentar esse rigor elevaria ainda mais a Precisão do modelo (reduzindo significativamente os falsos alarmes), mas sacrificaria o Recall (diminuindo o volume total de faltas identificadas). Portanto, a calibração do sistema na prática dependerá de uma decisão da instituição: aceitar um volume controlado de falsos positivos para combater ativamente a ociosidade médica, ou priorizar a segurança absoluta da recepção contra eventuais superlotações.

Para o escopo deste projeto, o comportamento atual alinha-se ao objetivo proposto: atuar como um **Sistema de Suporte à Decisão**. Com base nessas previsões, a gestão clínica ganha uma ferramenta de dados para fundamentar estratégias de mitigação da ociosidade — como o contato direcionado para confirmação de consultas da lista de risco —, mantendo sob controle os riscos associados aos encaixes incorretos.

# Pipeline de pesquisa e análise de dados

O desenvolvimento desta solução seguiu um pipeline estruturado para orientar a extração de informações e a construção do modelo. As etapas implementadas foram:

1. **Entendimento do Problema:** Definição da questão de pesquisa voltada à análise do absenteísmo em consultas e suas potenciais consequências na ociosidade de agendas médicas.
2. **Coleta e Análise Exploratória (EDA):** Limpeza dos dados, mapeamento demográfico do *dataset* e identificação do tempo de espera como o fator de maior influência para o status da consulta nos registros analisados.
3. **Preparação de Dados:** Limpeza de registros impossíveis, engenharia da variável `DiasEspera`, codificação numérica, separação agrupada por paciente (`GroupShuffleSplit` por `PatientId`, 70%/30%) anti-vazamento e padronização escalar.
4. **Modelagem Preditiva:** Avaliação de diferentes algoritmos (Baseline), seleção do Random Forest e otimização de seus hiperparâmetros via validação cruzada (`GridSearchCV`).
5. **Avaliação e Síntese:** Calibração do limiar de classificação (70%) com foco no controle de Falsos Positivos da Matriz de Confusão, direcionando a ferramenta para um uso assistido (*Human-in-the-Loop*).

## Observações importantes

Todas as tarefas, experimentações e testes descritos nesta etapa encontram-se registrados em formato de *notebook* (células de texto e código). O código-fonte desenvolvido para a execução deste pipeline está disponível na íntegra na pasta "src" do nosso repositório.
