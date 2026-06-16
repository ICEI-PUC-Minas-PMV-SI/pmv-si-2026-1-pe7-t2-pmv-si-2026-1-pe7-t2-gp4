# Construção de Modelos — Etapa 4

> Tema: Eficiência no SUS — estratégia de *Overbooking* Responsável a partir da previsão de *no-shows*.

Esta etapa estende a modelagem da Etapa 3 em cinco frentes: (1) ajustes de pré-processamento exigidos pelos novos algoritmos; (2) implementação de **dois novos algoritmos** — **XGBoost** e **LightGBM**; (3) avaliação com múltiplas métricas e definição de uma **métrica principal**; (4) comparação dos modelos pela métrica principal, com análise crítica; e (5) **refinamento e generalização do pipeline** em funções modulares, fechando com a **regra operacional de overbooking**.

As correções de pré-processamento, separação de dados, hiperparâmetros e nomenclatura levantadas no *feedback* da Etapa 3 estão documentadas em [construindo-modelo.md](construindo-modelo.md). Esta etapa também encerra dois pontos que ficaram em aberto: a **análise formal de *threshold tuning*** (tabela sistemática por limiar) e a **regra operacional de overbooking** (o pipeline antes parava na previsão de risco).

O código-fonte completo está em `src/Colab_4_etapa.ipynb`, na seção *"Etapa 4 — Novos Algoritmos, Pipeline Modular e Decisão Operacional"*.

---

# Preparação dos dados

Os dois novos algoritmos são baseados em **árvores de decisão impulsionadas por gradiente** (*gradient boosting*). Por serem modelos de árvore, herdam as mesmas propriedades da Random Forest da Etapa 3, o que define o que muda e o que se mantém.

**O que se mantém da Etapa 3 (sem alteração):**

* O mesmo conjunto de **9 *features*** (`Age`, `DiasEspera`, `Gender_numeric`, `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `SMS_received`) e a mesma variável-alvo `NoShow_numeric`.
* A separação **`GroupShuffleSplit` por `PatientId`**, preservando a estratégia anti-vazamento por paciente (`test_size=0.30`, `random_state=42`), resultando em **77.431 registros de treino e 33.091 de teste**, com taxa de *no-show* de ≈ 20,2% em ambos os conjuntos.
* A validação cruzada **agrupada (`GroupKFold`)** para qualquer busca de hiperparâmetros.

**O que muda em relação à Etapa 3:**

* **Escalonamento dispensado.** XGBoost e LightGBM são invariantes à escala das *features* (assim como a Random Forest). O `StandardScaler` — necessário apenas à Regressão Logística do *baseline* — deixa de ser aplicado. O pipeline torna o escalonamento **opcional** (parâmetro `escalonar`).
* **Desbalanceamento via `scale_pos_weight`.** Em vez do dicionário `class_weight={0:1, 1:2.5}` da Etapa 3, os modelos de *boosting* recebem `scale_pos_weight = nº negativos / nº positivos ≈ 3,957`, calculado **somente no conjunto de treino**. É a forma nativa e estatisticamente fundamentada de reponderar a classe minoritária nesses algoritmos.
* **Probabilidades como produto central.** A saída `predict_proba` passa a ser o produto principal do modelo (e não a classe binária), pois é a probabilidade que alimenta a regra de overbooking. Isso motiva a escolha da métrica principal (seção *Avaliação*).

Nenhuma nova limpeza de dados foi necessária: a base já tratada na EDA atende aos requisitos dos novos algoritmos.

# Descrição dos modelos

A seleção partiu das **características do problema**: dados **tabulares**, alvo **binário e desbalanceado** (~20% de *no-show*), presença de **interações não-lineares** entre variáveis (ex.: `DiasEspera` × `SMS_received`) e necessidade de **probabilidades de boa qualidade** para alimentar a regra de overbooking. Esse perfil favorece fortemente os métodos de *gradient boosting*, hoje considerados estado da arte para classificação tabular.

## Algoritmo 1 — XGBoost (Extreme Gradient Boosting)

Constrói árvores de forma **sequencial**: cada nova árvore corrige os erros residuais das anteriores — em contraste com a Random Forest da Etapa 3, que é *bagging* (árvores independentes em paralelo).

* **Vantagens para o caso:** tratamento nativo do desbalanceamento via `scale_pos_weight`; regularização L1/L2 que controla o sobreajuste; probabilidades bem ordenadas, ideais para a decisão de encaixe.
* **Limitações:** maior número de hiperparâmetros a calibrar (custo de *tuning*) e risco de sobreajuste se mal regularizado.
* **Hiperparâmetros usados:** `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight≈3.957`, `eval_metric="aucpr"`, `tree_method="hist"`.

## Algoritmo 2 — LightGBM (Light Gradient Boosting Machine)

Também é *boosting* de árvores, mas com duas diferenças algorítmicas centrais: crescimento **leaf-wise** (expande a folha de maior ganho, em vez de nível a nível) e **binning por histograma** das *features*. O resultado é treino muito mais rápido e menor uso de memória, mantendo acurácia competitiva.

* **Vantagens para o caso:** **escalabilidade** — o *dataset* tem ~110 mil registros, mas uma implantação real no SUS lidaria com milhões; o LightGBM viabiliza o reuso do pipeline nessa escala.
* **Limitações:** o crescimento *leaf-wise* pode sobreajustar em bases pequenas (não é o caso aqui).
* **Hiperparâmetros usados:** `n_estimators=300`, `num_leaves=31`, `max_depth=-1`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight≈3.957`.

A Random Forest da Etapa 3 é mantida como **incumbente/referência**, reutilizando os melhores hiperparâmetros do `GridSearchCV` (`max_depth=10`, `min_samples_leaf=4`, `min_samples_split=10`, `n_estimators=100`, `class_weight="balanced"`). Comparar **Random Forest (bagging)** × **XGBoost (boosting nível/histograma)** × **LightGBM (boosting leaf-wise)** cobre, portanto, três estratégias distintas de *ensemble* de árvores, permitindo avaliar se a forma de combinar as árvores muda o desempenho neste problema.

# Avaliação dos modelos criados

## Métricas utilizadas

Foram avaliadas múltiplas métricas (acurácia, precisão, *recall*, F1, ROC-AUC e PR-AUC), mas, conforme exigido, definimos **uma métrica principal** que guia a seleção e a comparação dos modelos.

**Métrica principal: PR-AUC** (área sob a curva Precisão–Recall, *Average Precision*). Justificativa, alinhada às especificidades do problema:

* **Desbalanceamento.** Com apenas ~20% de *no-shows*, a **acurácia é enganosa** (basta prever "compareceu" para todos para acertar ~80%) e a **ROC-AUC é otimista**, pois incorpora os verdadeiros negativos — abundantes e fáceis. A PR-AUC ignora os verdadeiros negativos e mede o desempenho exatamente sobre a **classe de interesse (a falta)**.
* **Independência de limiar.** A entrega final é orientada a **probabilidades**: a regra de overbooking consome o *ranking* de risco, não uma decisão binária fixa. A PR-AUC avalia a qualidade desse ranqueamento em todos os limiares de uma só vez, medindo a capacidade **intrínseca** do modelo — algo que uma métrica presa a um único limiar (como o F1 em 0,50) não captura.
* **Conexão direta com o *threshold tuning*.** Separamos duas decisões: (i) **escolher o modelo** pela PR-AUC (capacidade de ranquear o risco) e (ii) **escolher o limiar operacional** pela tabela de custo. Isso responde diretamente ao *feedback* da Etapa 3, que pedia que a métrica principal estivesse ligada à escolha do limiar.

As demais métricas (F1, precisão, *recall* no limiar operacional, ROC-AUC) permanecem como **métricas secundárias** de apoio à leitura de negócio.

## Comparação pela métrica principal

Resultados no conjunto de teste (33.091 consultas), todos os modelos com `random_state=42` e limiar de 0,50 para as métricas dependentes de classe:

| Modelo | PR-AUC (principal) | ROC-AUC | F1 (0,50) | Recall (0,50) | Precisão (0,50) |
|---|---:|---:|---:|---:|---:|
| Random Forest (Etapa 3) | **0,3512** | 0,7234 | 0,4419 | 0,8038 | 0,3047 |
| **XGBoost** | 0,3505 | **0,7244** | **0,4421** | 0,7996 | 0,3055 |
| LightGBM | 0,3467 | 0,7224 | 0,4406 | 0,7928 | 0,3051 |

> Os valores podem variar na 3ª/4ª casa decimal conforme a versão das bibliotecas; o ranqueamento e as conclusões se mantêm.

## Discussão dos resultados obtidos

A diferença de **PR-AUC entre os três modelos é mínima** (todos na faixa de ~0,35) — esse é, por si só, o achado mais importante da comparação. Na execução registrada, a Random Forest fica marginalmente à frente na métrica principal (0,3512 vs. 0,3505 do XGBoost, uma diferença na 3ª casa decimal, dentro do ruído estatístico), enquanto o **XGBoost lidera ROC-AUC e F1** e produz as melhores probabilidades.

* **Random Forest (incumbente).** Robusta, estável e a mais **interpretável** (a *feature importance* da Etapa 3 mostra `DiasEspera` com ~76% do peso). Limitação: como *bagging*, atinge um teto de desempenho e não extrai sinal adicional sutil.
* **XGBoost.** Melhor ROC-AUC e F1 (por margem estreita) e probabilidades bem calibradas — vantagem direta para a regra de overbooking. Limitação: mais hiperparâmetros a calibrar e risco de sobreajuste se mal regularizado.
* **LightGBM.** Desempenho **praticamente idêntico** ao do XGBoost, porém com treino sensivelmente **mais rápido** — vantagem decisiva para a escala de produção do SUS. Limitação: *leaf-wise* pode sobreajustar em bases pequenas (não é o caso aqui).

**Conclusão crítica.** Trocar a Random Forest por *boosting* traz ganho **marginal**. Isso indica que o **teto preditivo está limitado pelas *features* disponíveis**, e não pelo algoritmo — uma única variável (`DiasEspera`) domina o sinal. O caminho mais promissor de evolução **não é trocar de modelo**, e sim **enriquecer as variáveis** (histórico de faltas anteriores do próprio paciente, distância casa–unidade, sazonalidade/dia da semana, nº de remarcações). Adotamos o **XGBoost como modelo campeão** desta etapa por oferecer as melhores probabilidades para a decisão operacional e liderar as métricas secundárias relevantes, mantendo o LightGBM como alternativa recomendada quando a escala exigir velocidade.

## Threshold tuning — análise formal de limiar

O *feedback* da Etapa 3 apontou que a escolha do limiar de 70% ficou **argumentativa**, sem uma tabela sistemática. A tabela abaixo (gerada pela função `analise_threshold`) corrige isso para o **modelo campeão (XGBoost)**, comparando precisão, *recall*, F1, falsos positivos/negativos e um **custo operacional ponderado**.

Adotamos `custo_fp = 1,5` e `custo_fn = 1,0`: um **falso positivo** (alertar uma falta que não ocorre → encaixe indevido → risco de superlotação) é considerado mais custoso que um **falso negativo** (falta não prevista → vaga ociosa). Esses pesos são parâmetros de negócio e podem ser recalibrados pela gestão.

| Threshold | Precisão | Recall | F1 | FP | FN | % alertas | Custo operacional |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,50 | 0,305 | 0,800 | 0,442 | 12.169 | 1.341 | 52,9% | 19.594,5 |
| 0,60 | 0,342 | 0,549 | 0,421 | 7.066 | 3.021 | 32,4% | 13.620,0 |
| 0,70 | 0,393 | **0,190** | **0,257** | 1.966 | 5.418 | 9,8% | 8.367,0 |
| 0,80 | 0,452 | 0,011 | 0,021 | 86 | 6.622 | 0,5% | 6.751,0 |

A tabela **confirma empiricamente** o que o *feedback* antecipou: em 0,70 o *recall* colapsa para ~0,19 e o F1 da classe positiva cai para ~0,26 — o modelo deixa de detectar a maioria das faltas. Em 0,50, recupera-se *recall* alto (~0,80) ao custo de muitos falsos positivos. **Não existe um limiar único "ótimo"**: é um *trade-off* explícito de negócio. Por isso, para o alerta individual de um paciente, a gestão escolhe a linha que melhor equilibra custo e *recall*; já para a decisão de capacidade (quantos encaixes liberar por agenda), usamos a abordagem da próxima seção, que **dispensa limiar**.

# Revisão do pipeline de pesquisa e análise de dados

O pipeline da Etapa 3 foi refatorado em **funções independentes, reutilizáveis e documentadas**. Cada função tem responsabilidade única e pode ser reaproveitada em outros contextos — trocar de problema exige apenas alterar `df`, a lista de *features* ou o dicionário de modelos. Essas funções são a espinha dorsal de toda a Etapa 4:

| Função | Responsabilidade |
|---|---|
| `preparar_dados(...)` | *Split* por grupo (paciente) anti-vazamento; escalonamento opcional; auditoria automática que falha se algum paciente vazar entre treino e teste |
| `avaliar_modelo(...)` | Treina o modelo e devolve todas as métricas + probabilidades (`predict_proba`) |
| `comparar_modelos(...)` | Tabela ordenada pela métrica principal (PR-AUC) |
| `analise_threshold(...)` | *Threshold tuning* com custo operacional ponderado |
| `politica_overbooking(...)` | Traduz probabilidades em nº de encaixes recomendados por agenda |

Em relação ao pipeline da Etapa 3, as principais mudanças foram: (i) **modularização** em funções reutilizáveis (antes o código era sequencial e específico); (ii) **escalonamento opcional**, refletindo que os modelos de árvore dispensam padronização; (iii) **PR-AUC como métrica de seleção**, no lugar do F1 em limiar fixo; e (iv) o **fechamento operacional** descrito a seguir, que faltava na etapa anterior.

## Fechamento do pipeline — regra operacional de overbooking

O *feedback* observou que o pipeline **parava na previsão de risco** e não traduzia a probabilidade em uma **regra de encaixe**. Esta seção fecha essa lacuna.

Para cada agenda (aqui aproximada por **unidade × dia**), o número esperado de faltas é a **soma das probabilidades** de falta dos pacientes agendados — o valor esperado de uma soma de variáveis de Bernoulli, um estimador que **dispensa limiar** e aproveita toda a informação do `predict_proba`. A recomendação de encaixes é:

> `encaixes = floor( soma(probabilidades) × fator_seguranca )`, **limitada** a `floor( tamanho_da_agenda × teto_pct )`.

* `fator_seguranca = 0,85` aplica uma **margem conservadora**: liberamos menos encaixes do que o número esperado de faltas, reduzindo o risco de superlotação (*overflow*).
* `teto_pct = 0,15` é um **limite duro de segurança**: nenhuma agenda recebe mais que 15% de encaixes extras, qualquer que seja a previsão.

Aplicada às **356 agendas** de teste com ≥ 20 consultas:

* Total de encaixes sugeridos: **1.442**; faltas reais nessas agendas: **2.153** → recuperação de **67,0%** da capacidade ociosa.
* *Overflow* (encaixes acima das faltas reais) ocorre em **14,6%** das agendas e, quando ocorre, é de **1,73 paciente** em média.

`fator_seguranca` e `teto_pct` são as **alavancas de gestão**: mais agressivo recupera mais vagas; mais conservador minimiza o risco de superlotação. Eticamente, o modelo é **suporte à decisão** (*human-in-the-loop*): informa risco **agregado por agenda**, sem cancelar consultas individuais nem usar atributos sociodemográficos de forma discriminatória.

## Observações importantes 

Todas as tarefas, experimentações e testes desta etapa estão registrados em formato de *notebook* (células de texto e código). O código-fonte desenvolvido está disponível na íntegra na pasta `src` do repositório (`src/Colab_4_etapa.ipynb`).
