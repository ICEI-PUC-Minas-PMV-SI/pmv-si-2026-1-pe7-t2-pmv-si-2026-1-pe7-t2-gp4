# Construção de Modelos — Etapa 4

> Tema: Eficiência no SUS — estratégia de *Overbooking* Responsável a partir da previsão de *no-shows*.

Esta etapa estende a modelagem da Etapa 3 em cinco frentes: (1) ajustes de pré-processamento exigidos pelos novos algoritmos; (2) implementação de **dois novos algoritmos** — **XGBoost** e **LightGBM**; (3) avaliação com múltiplas métricas — incluindo **métricas de calibração** — e definição de uma **métrica principal**; (4) comparação dos modelos pela métrica principal, com análise crítica; e (5) **refinamento e generalização do pipeline** em funções modulares, fechando com a **regra operacional de overbooking**.

As correções de pré-processamento, separação de dados, hiperparâmetros e nomenclatura levantadas no *feedback* da Etapa 3 estão documentadas em [construindo-modelo.md](construindo-modelo.md). Esta etapa também encerra dois pontos que ficaram em aberto: a **análise formal de *threshold tuning*** (tabela sistemática por limiar) e a **regra operacional de overbooking** (o pipeline antes parava na previsão de risco).

O código-fonte completo está em `src/Colab_5_etapa.ipynb`, na seção *"Etapa 4 — Novos Algoritmos, Pipeline Modular e Decisão Operacional"*.

---

# Preparação dos dados

Os dois novos algoritmos são baseados em **árvores de decisão impulsionadas por gradiente** (*gradient boosting*). Por serem modelos de árvore, herdam as mesmas propriedades da Random Forest da Etapa 3, o que define o que muda e o que se mantém.

**O que se mantém da Etapa 3 (sem alteração):**

* O mesmo conjunto de **9 *features*** (`Age`, `DiasEspera`, `Gender_numeric`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `SMS_received`) e a mesma variável-alvo `NoShow_numeric`.
* A separação **por grupo (`PatientId`)** via `GroupShuffleSplit`, preservando a estratégia anti-vazamento por paciente, e a validação cruzada **agrupada (`GroupKFold`)** para qualquer busca de hiperparâmetros.

**O que muda em relação à Etapa 3:**

* **Separação em três conjuntos (Treino / Validação / Teste).** Em vez do *split* binário treino/teste da Etapa 3, a função `preparar_dados_cv` isola primeiro o **Teste** (`test_size=0.20`) e depois separa **Treino** e **Validação** (`val_size=0.20` do restante), sempre agrupando por `PatientId` (`random_state=42`). O resultado são **≈ 70.782 registros de treino, ≈ 17.659 de validação e ≈ 22.080 de teste**, com taxa de *no-show* de ≈ 20% nos três conjuntos. A seleção de modelo e o *threshold tuning* são feitos **na validação**; o conjunto de **teste é mantido intocado** e usado apenas na **auditoria final**, evitando vazamento por otimização.
* **Escalonamento dispensado.** XGBoost e LightGBM são invariantes à escala das *features* (assim como a Random Forest). O `StandardScaler` — necessário apenas à Regressão Logística do *baseline* — deixa de ser aplicado. O pipeline torna o escalonamento **opcional** (parâmetro `escalonar`).
* **Desbalanceamento via `scale_pos_weight`.** Em vez do dicionário `class_weight={0:1, 1:2.5}` da Etapa 3, os modelos de *boosting* recebem `scale_pos_weight = nº negativos / nº positivos ≈ 3,96`, calculado **somente no conjunto de treino**. É a forma nativa e estatisticamente fundamentada de reponderar a classe minoritária nesses algoritmos.
* **Probabilidades como produto central.** A saída `predict_proba` passa a ser o produto principal do modelo (e não a classe binária), pois é a probabilidade que alimenta a regra de overbooking. Isso motiva a escolha da métrica principal (seção *Avaliação*).

Nenhuma nova limpeza de dados foi necessária: a base já tratada na EDA atende aos requisitos dos novos algoritmos.

# Descrição dos modelos

A seleção partiu das **características do problema**: dados **tabulares**, alvo **binário e desbalanceado** (~20% de *no-show*), presença de **interações não-lineares** entre variáveis (ex.: `DiasEspera` × `SMS_received`) e necessidade de **probabilidades de boa qualidade** para alimentar a regra de overbooking. Esse perfil favorece fortemente os métodos de *gradient boosting*, hoje considerados estado da arte para classificação tabular.

## Algoritmo 1 — XGBoost (Extreme Gradient Boosting)

Constrói árvores de forma **sequencial**: cada nova árvore corrige os erros residuais das anteriores — em contraste com a Random Forest da Etapa 3, que é *bagging* (árvores independentes em paralelo).

* **Vantagens para o caso:** tratamento nativo do desbalanceamento via `scale_pos_weight`; regularização L1/L2 que controla o sobreajuste; probabilidades bem ordenadas, ideais para a decisão de encaixe.
* **Limitações:** maior número de hiperparâmetros a calibrar (custo de *tuning*) e risco de sobreajuste se mal regularizado.
* **Hiperparâmetros:** ajustados por **`RandomizedSearchCV`** (`n_iter=5`) com `GroupKFold(n_splits=3)` e *score* na **PR-AUC**, varrendo `max_depth ∈ {3,5,7}`, `learning_rate ∈ {0,01; 0,05; 0,1}` e `n_estimators ∈ {100, 200}`. Parâmetros fixos: `scale_pos_weight ≈ 3,96`, `eval_metric="logloss"`, `tree_method="hist"`.

## Algoritmo 2 — LightGBM (Light Gradient Boosting Machine)

Também é *boosting* de árvores, mas com duas diferenças algorítmicas centrais: crescimento **leaf-wise** (expande a folha de maior ganho, em vez de nível a nível) e **binning por histograma** das *features*. O resultado é treino muito mais rápido e menor uso de memória, mantendo acurácia competitiva.

* **Vantagens para o caso:** **escalabilidade** — o *dataset* tem ~110 mil registros, mas uma implantação real no SUS lidaria com milhões; o LightGBM viabiliza o reuso do pipeline nessa escala.
* **Limitações:** o crescimento *leaf-wise* pode sobreajustar em bases pequenas (não é o caso aqui).
* **Hiperparâmetros:** treinado com a configuração **padrão** do `LGBMClassifier` (`num_leaves=31`, `max_depth=-1`, `n_estimators=100`, `learning_rate=0.1`), acrescido apenas de `scale_pos_weight ≈ 3,96` para o desbalanceamento.

A Random Forest da Etapa 3 é mantida como **incumbente/referência**, mas **retreinada no *split* desta etapa** com `class_weight="balanced"`. Reaproveitar o modelo da Etapa 3 (ajustado em outro *split*) causaria vazamento de dados na avaliação. Comparar **Random Forest (bagging)** × **XGBoost (boosting sequencial/histograma)** × **LightGBM (boosting leaf-wise)** cobre, portanto, três estratégias distintas de *ensemble* de árvores, permitindo avaliar se a forma de combinar as árvores muda o desempenho neste problema.

# Avaliação dos modelos criados

## Métricas utilizadas

Foram avaliadas múltiplas métricas (acurácia, precisão, *recall*, F1, ROC-AUC, PR-AUC) e, adicionalmente, **métricas de calibração** — **Brier Score** e **Log Loss** (em ambas, **menor é melhor**) —, que medem o quão confiáveis são as probabilidades, e não apenas a ordenação. Conforme exigido, definimos **uma métrica principal** que guia a seleção e a comparação dos modelos.

**Métrica principal: PR-AUC** (área sob a curva Precisão–Recall, *Average Precision*). Justificativa, alinhada às especificidades do problema:

* **Desbalanceamento.** Com apenas ~20% de *no-shows*, a **acurácia é enganosa** (basta prever "compareceu" para todos para acertar ~80%) e a **ROC-AUC é otimista**, pois incorpora os verdadeiros negativos — abundantes e fáceis. A PR-AUC ignora os verdadeiros negativos e mede o desempenho exatamente sobre a **classe de interesse (a falta)**.
* **Independência de limiar.** A entrega final é orientada a **probabilidades**: a regra de overbooking consome o *ranking* de risco, não uma decisão binária fixa. A PR-AUC avalia a qualidade desse ranqueamento em todos os limiares de uma só vez, medindo a capacidade **intrínseca** do modelo — algo que uma métrica presa a um único limiar (como o F1 em 0,50) não captura.
* **Conexão direta com o *threshold tuning*.** Separamos duas decisões: (i) **escolher o modelo** pela PR-AUC (capacidade de ranquear o risco) e (ii) **escolher o limiar operacional** pela tabela de custo. Isso responde diretamente ao *feedback* da Etapa 3, que pedia que a métrica principal estivesse ligada à escolha do limiar.

As demais métricas (F1, precisão, *recall* no limiar operacional, ROC-AUC, Brier e Log Loss) permanecem como **métricas secundárias** de apoio à leitura de negócio e à verificação da calibração.

## Comparação pela métrica principal

Resultados no conjunto de **validação** (≈ 17.659 consultas), todos os modelos com `random_state=42` e limiar de 0,50 para as métricas dependentes de classe. Em **Brier** e **Log Loss**, menor é melhor:

| Modelo | PR-AUC (principal) | Brier ↓ | Log Loss ↓ | ROC-AUC | F1 (0,50) | Recall (0,50) |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM (campeão)** | **~0,35** | ~0,21 | **~0,60** | **~0,73** | ~0,44 | ~0,81 |
| XGBoost (tunado) | ~0,33 | ~0,217 | ~0,62 | ~0,72 | ~0,44 | **~0,88** |
| Random Forest (retreinada) | ~0,27 | ~0,202 | ~0,93 | ~0,62 | ~0,36 | ~0,48 |

> Os valores exatos variam conforme a versão das bibliotecas (XGBoost, LightGBM, scikit-learn) e a amostragem da `RandomizedSearchCV`. A seleção do modelo campeão segue a **PR-AUC registrada na execução do notebook**, onde o LightGBM lidera a métrica principal; XGBoost e LightGBM ficam tecnicamente empatados, e a Random Forest fica nitidamente atrás.

## Discussão dos resultados obtidos

Os dois algoritmos de *boosting* ficaram **tecnicamente empatados na métrica principal** (PR-AUC ≈ 0,33–0,35 na validação), enquanto a **Random Forest ficou nitidamente atrás** (PR-AUC ≈ 0,27 e ROC-AUC ≈ 0,62, próximo do acaso). Esse é o achado central da comparação: aqui o *boosting* supera o *bagging* com folga na capacidade de ordenar o risco.

* **LightGBM (campeão).** Lidera a **métrica principal (PR-AUC ≈ 0,35)** e ainda apresenta o **menor Log Loss** (≈ 0,60) e a melhor **ROC-AUC** (≈ 0,73), com Brier praticamente empatado com o melhor. Soma a isso o treino **mais rápido** (*leaf-wise* + *binning* por histograma) — vantagem decisiva para a escala de produção do SUS. Limitação: o crescimento *leaf-wise* pode sobreajustar em bases pequenas (não é o caso aqui).
* **XGBoost.** Desempenho muito próximo do LightGBM e o **maior recall** (≈ 0,88) — captura mais faltas no limiar de 0,50. Limitações: fica **atrás na PR-AUC** (≈ 0,33), tem o **pior Brier** dos três (≈ 0,217) e maior custo de *tuning* (mais hiperparâmetros).
* **Random Forest (incumbente da Etapa 3).** É a mais **interpretável** (a *feature importance* mostra `DiasEspera` com ~76% do peso). Porém, retreinada no *split* desta etapa, **discrimina mal o risco**: a pior PR-AUC e ROC-AUC ≈ 0,62. Atenção a uma armadilha de leitura: seu **Brier é o menor da tabela (≈ 0,202)**, mas isso não significa boa calibração útil — combinado com Log Loss alto (≈ 0,93) e ROC-AUC baixa, é o padrão de um modelo que empurra quase todas as probabilidades para perto da taxa-base (~20%). Ele acerta o "nível médio", mas não ordena quem vai faltar de quem vai comparecer — exatamente o que a política de overbooking precisa.

**Conclusão crítica.** Seguindo a regra metodológica (seleção pela **métrica principal, PR-AUC**), o **LightGBM é declarado modelo campeão**: não vence apenas a PR-AUC — também lidera ROC-AUC e Log Loss, empata no topo do Brier e iguala o XGBoost em F1, somando ainda a vantagem de escalabilidade. A pequena perda de *recall* frente ao XGBoost é uma métrica secundária, gerenciada no *threshold tuning*. A Random Forest fica descartada por discriminar o risco de forma insuficiente. Como o ganho relevante vem de trocar *bagging* por *boosting* — e não de aprofundar o *boosting* —, o **teto preditivo passa a ser limitado pelas *features* disponíveis** (uma única variável, `DiasEspera`, domina o sinal). O caminho mais promissor de evolução **não é trocar de modelo**, e sim **enriquecer as variáveis** (histórico de faltas anteriores do próprio paciente, distância casa–unidade, sazonalidade/dia da semana, nº de remarcações).

## Threshold tuning — análise formal de limiar

O *feedback* da Etapa 3 apontou que a escolha do limiar de 70% ficou **argumentativa**, sem uma tabela sistemática. A tabela abaixo (gerada pela função `analise_threshold` **sobre a validação**) corrige isso para o **modelo campeão (LightGBM)**, comparando precisão, *recall*, F1, falsos positivos/negativos e um **custo operacional ponderado**.

Adotamos `custo_fp = 1,5` e `custo_fn = 1,0`: um **falso positivo** (alertar uma falta que não ocorre → encaixe indevido → risco de superlotação) é considerado mais custoso que um **falso negativo** (falta não prevista → vaga ociosa). Esses pesos são parâmetros de negócio e podem ser recalibrados pela gestão.

| Threshold | Precisão | Recall | F1 | FP | FN | Custo operacional |
|---:|---:|---:|---:|---:|---:|---:|
| 0,50 | 0,302 | **0,807** | 0,440 | 6.558 | 680 | 10.517,0 |
| 0,60 | 0,351 | 0,556 | 0,430 | 3.618 | 1.566 | 6.993,0 |
| 0,70 | 0,392 | **0,189** | **0,255** | 1.034 | 2.857 | 4.408,0 |
| 0,80 | 0,483 | 0,008 | 0,016 | 31 | 3.495 | 3.541,5 |

A tabela **confirma empiricamente** o que o *feedback* antecipou: em 0,70 o *recall* colapsa para ~0,19 e o F1 da classe positiva cai para ~0,26 — o modelo deixa de detectar a maioria das faltas. Em 0,50, recupera-se *recall* alto (~0,81) ao custo de muitos falsos positivos. **Não existe um limiar único "ótimo"**: é um *trade-off* explícito de negócio. Por isso, para o alerta individual de um paciente, a gestão escolhe a linha que melhor equilibra custo e *recall*; já para a decisão de capacidade (quantos encaixes liberar por agenda), usamos a abordagem da próxima seção, que **dispensa limiar**.

# Revisão do pipeline de pesquisa e análise de dados

O pipeline da Etapa 3 foi refatorado em **funções independentes, reutilizáveis e documentadas**, exportadas para um módulo Python próprio (`pipeline.py`). Cada função tem responsabilidade única e pode ser reaproveitada em outros contextos — trocar de problema exige apenas alterar `df`, a lista de *features* ou o dicionário de modelos. Essas funções são a espinha dorsal de toda a Etapa 4:

| Função | Responsabilidade |
|---|---|
| `preparar_dados_cv(...)` | *Split* por grupo (paciente) em **Treino / Validação / Teste**, anti-vazamento; escalonamento opcional |
| `avaliar_modelo_completo(...)` | Avalia o modelo e devolve métricas de classificação **e de calibração** (PR-AUC, Brier, Log Loss, ROC-AUC, F1, precisão, recall) + as probabilidades (`predict_proba`) |
| `comparar_modelos(...)` | Tabela ordenada pela métrica principal (PR-AUC) |
| `analise_threshold(...)` | *Threshold tuning* com custo operacional ponderado |
| `politica_overbooking(...)` | Traduz probabilidades em nº de encaixes recomendados por agenda |

Em relação ao pipeline da Etapa 3, as principais mudanças foram: (i) **modularização** em funções reutilizáveis exportadas para `pipeline.py` (antes o código era sequencial e específico); (ii) **separação em três conjuntos** (Treino/Validação/Teste), com seleção e *tuning* na validação e o teste reservado para auditoria final; (iii) **escalonamento opcional**, refletindo que os modelos de árvore dispensam padronização; (iv) **PR-AUC como métrica de seleção** (mais Brier/Log Loss como provas de calibração), no lugar do F1 em limiar fixo; e (v) o **fechamento operacional** descrito a seguir, que faltava na etapa anterior.

## Fechamento do pipeline — regra operacional de overbooking

O *feedback* observou que o pipeline **parava na previsão de risco** e não traduzia a probabilidade em uma **regra de encaixe**. Esta seção fecha essa lacuna, aplicando o modelo campeão ao **conjunto de teste** (intocado até aqui).

Para cada agenda (aqui aproximada por **unidade × dia**), o número esperado de faltas é a **soma das probabilidades** de falta dos pacientes agendados — o valor esperado de uma soma de variáveis de Bernoulli, um estimador que **dispensa limiar** e aproveita toda a informação do `predict_proba`. A recomendação de encaixes é:

> `encaixes = floor( soma(probabilidades) × fator_seguranca )`, **limitada** a `floor( tamanho_da_agenda × teto_pct )`.

* `fator_seguranca = 0,85` aplica uma **margem conservadora**: liberamos menos encaixes do que o número esperado de faltas, reduzindo o risco de superlotação (*overflow*).
* `teto_pct = 0,15` é um **limite duro de segurança**: nenhuma agenda recebe mais que 15% de encaixes extras, qualquer que seja a previsão.

Aplicada às **356 agendas** de teste com ≥ 20 consultas:

* Total de encaixes sugeridos: **1.442**; faltas reais nessas agendas: **2.153** → recuperação de **67,0%** da capacidade ociosa.
* *Overflow* (encaixes acima das faltas reais) ocorre em **14,6%** das agendas e, quando ocorre, é de **1,73 paciente** em média.

`fator_seguranca` e `teto_pct` são as **alavancas de gestão**: mais agressivo recupera mais vagas; mais conservador minimiza o risco de superlotação. Eticamente, o modelo é **suporte à decisão** (*human-in-the-loop*): informa risco **agregado por agenda**, sem cancelar consultas individuais nem usar atributos sociodemográficos de forma discriminatória.

## Conformidade com a LGPD e ética em pesquisa

Em conformidade com a **LGPD (Lei nº 13.709/2018)**, a análise respeita os princípios de **adequação e minimização**. O *dataset* é anonimizado e o `PatientId` é usado estritamente como **chave técnica** (`GroupShuffleSplit`/`GroupKFold`) para impedir o vazamento de dados — nunca como variável preditora. Variáveis médicas (como `Hipertension` e `Diabetes`) são tratadas como **dados pessoais sensíveis**. A finalidade do processamento e da regra operacional é **exclusivamente administrativa e acadêmica** (combate à ociosidade do SUS), sendo **proibida** a utilização dessas previsões para restringir o direito ao atendimento dos cidadãos com base em perfis preditivos.

## Observações importantes

Todas as tarefas, experimentações e testes desta etapa estão registrados em formato de *notebook* (células de texto e código). O código-fonte desenvolvido está disponível na íntegra na pasta `src` do repositório (`src/Colab_5_etapa.ipynb`), e as funções modulares do pipeline foram exportadas para `pipeline.py`.
