# Introdução

Este projeto apresenta uma proposta de otimização para a gestão de agendas no Sistema Único de Saúde (SUS), focando no absenteísmo em consultas especializadas no estado de **Minas Gerais**. Utilizando dados oficiais do **DataSUS (SIA/SUS) a partir do ano de 2020**, o trabalho propõe a aplicação de modelos de Aprendizado de Máquina para fundamentar uma estratégia de **overbooking dinâmico**. O objetivo é maximizar a ocupação das agendas médicas, garantindo que ociosidades previstas em um contexto pós-pandêmico sejam compensadas por agendamentos excedentes, acelerando o atendimento da população mineira.

## Problema

O problema central é a ineficiência causada pelo absenteísmo em consultas agendadas, que gera ociosidade de médicos e infraestrutura em Minas Gerais.

- **Contexto Regional:** O estado possui uma vasta rede de atendimento, mas sofre com disparidades regionais e filas de espera que se acumularam após o período crítico da pandemia de 2020.

- **Impacto:** Horários vagos que poderiam ser utilizados por pacientes em filas de espera de especialidades críticas (como cardiologia e oncologia).

- **Foco Temporal:** A partir de 2020, o comportamento de comparecimento mudou drasticamente devido à telemedicina e aos novos protocolos de saúde, tornando dados antigos obsoletos para predição atual.

## Questão de pesquisa

Como modelos de aprendizado de máquina podem fundamentar uma política de overbooking inteligente nas unidades de saúde de Minas Gerais, considerando as mudanças no perfil de comparecimento dos pacientes desde 2020?

## Objetivos preliminares

O objetivo geral é experimentar modelos de aprendizado de máquina adequados para prever o risco de no-show no cenário mineiro pós-2020 e propor uma gestão de agenda resiliente.

### Objetivos específicos:

- Extrair e filtrar dados do SIA/SUS (DataSUS) focados no estado de Minas Gerais (2020-2026);

- Comparar o desempenho de algoritmos de classificação para identificar o "padrão de falta" do paciente mineiro atual;

- Desenvolver um modelo de overbooking que ajuste a oferta de vagas extras conforme o risco sazonal e regional detectado nos dados.

## Justificativa

A escolha de dados do DataSUS justifica-se pela soberania e fidelidade das informações oficiais do governo brasileiro.

- **Relevância:** Focar em MG permite uma análise direcionada às necessidades locais, como a logística de transporte intermunicipal de pacientes.

- **Impacto:** Dados pós-2020 refletem a realidade atual do sistema de saúde pós-crise sanitária. Minimizar a ociosidade médica em Minas Gerais tem um impacto direto na redução da fila de espera de exames e consultas de alta complexidade.

## Público-Alvo

- **Secretaria de Estado de Saúde de Minas Gerais (SES-MG) e Gestores Municipais:** Decisores que precisam otimizar o uso do teto financeiro e aumentar a produtividade das unidades.

- **Profissionais de Regulação de Minas:** Usuários que operam o agendamento de consultas entre municípios (Consórcios de Saúde), buscando reduzir o impacto de faltas em viagens de pacientes.

## Estado da arte

Nesta seção, descrevemos abordagens da literatura que utilizam Ciência de Dados para enfrentar o absenteísmo na saúde pública, com foco em dados brasileiros e métodos de otimização de agendas.

### **1. Predição de No-Show em Minas Gerais via SIA/SUS**

**Problema e contexto:** Investigar o impacto da pandemia de COVID-19 no absenteísmo em consultas especializadas em municípios de Minas Gerais.

**Dados:** Microdados do SIA/SUS (DataSUS), período 2020-2022. Variáveis incluem idade, sexo, especialidade médica e distância do município de residência.

**Abordagem/algoritmos:** Random Forest e Regressão Logística.

**Métricas de avaliação:** Acurácia e AUC-ROC (para medir a capacidade de distinção entre quem vai e quem falta).

**Resultados:** O modelo Random Forest atingiu AUC de 0.82. Concluiu-se que o represamento de demandas pós-2020 alterou o perfil de faltas, aumentando o absenteísmo em consultas agendadas com mais de 60 dias de antecedência.

### **2. Otimização de Agendas via Overbooking Probabilístico**

**Problema e contexto:** Resolver a ociosidade médica em clínicas ambulatoriais através de um sistema de "agendamento excedente" baseado no risco individual do paciente.

**Dados:** 85.000 registros de agendamentos eletrônicos. Pré-processamento: Normalização de idade e codificação (One-Hot Encoding) de especialidades.

**Abordagem/algoritmos:** XGBoost para predição e Simulação de Monte Carlo para determinar o limite de overbooking.

**Métricas de avaliação:** Taxa de Ocupação da Agenda e Tempo Médio de Espera (MAE).

**Resultados:** Redução de 22% na ociosidade médica com um aumento marginal de apenas 5 minutos no tempo de espera em sala, validando a estratégia de overbooking.

### **3. O Absenteísmo Pós-Pandemia: Uma Análise do SIA/SUS**

**Problema e contexto:** Identificar se fatores socioeconômicos pós-2020 tornaram-se preditores mais fortes de no-show.

**Dados:** DataSUS (SIA), foco na Região Sudeste, 2021-2023. Atributos: Caráter do atendimento, código do procedimento e ID do prestador.

**Abordagem/algoritmos:** Naive Bayes e K-Nearest Neighbors (KNN).

**Métricas de avaliação:** F1-Score (devido ao desbalanceamento das classes, onde há mais presenças que faltas).

**Resultados:** O Naive Bayes mostrou-se eficiente para triagem rápida, identificando que procedimentos de alta complexidade em MG têm menor taxa de falta que consultas básicas.

### **4. Impacto dos Consórcios Intermunicipais de Saúde em MG**

**Problema e contexto:** O absenteísmo no transporte de pacientes entre cidades pequenas de MG e polos de saúde.

**Dados:** Registros administrativos de Consórcios Intermunicipais de Saúde (2020-2024).

**Abordagem/algoritmos:** Árvores de Decisão (C4.5).

**Métricas de avaliação:** Precisão e Recall.

**Resultados:** Demonstrou que a falta de confirmação em 48h antes da viagem é o principal preditor de falta, sugerindo que o overbooking deveria ser aplicado especificamente para pacientes locais.

### **5. Uso de BERTimbau para Análise de Motivação de Faltas**

**Problema e contexto:** Classificar justificativas textuais de pacientes para faltas em exames e consultas.

**Dados:** Logs de sistemas de atendimento e mensagens de texto (anonimizados).

**Abordagem/algoritmos:** LLM (BERTimbau - BERT para português).

**Métricas de avaliação:** Acurácia e Matriz de Confusão.

**Resultados:** Alcançou 89% de precisão na classificação de motivos (transporte, esquecimento, melhora do sintoma), servindo de base para ajustar pesos no modelo de predição de overbooking.

## Texto-síntese Crítico
Os estudos analisados concordam que o absenteísmo é um problema multifatorial e que o uso de modelos preditivos supera amplamente a gestão intuitiva de agendas. Há um consenso de que variáveis temporais (tempo de espera entre marcação e consulta) e geográficas (deslocamento do paciente) são os preditores mais robustos no cenário brasileiro. Contudo, divergem quanto ao algoritmo ideal: enquanto modelos de árvore (Random Forest/XGBoost) apresentam maior acurácia para dados tabulares, modelos estatísticos simples como Naive Bayes oferecem maior transparência para implementação em sistemas públicos legados.

As principais lacunas identificadas residem na ausência de aplicações práticas que integrem a predição diretamente com a ação de overbooking automático no SUS. A maioria dos estudos limita-se a "prever o problema", mas poucos propõem a "solução operacional" baseada em margens de segurança para evitar superlotação. Além disso, há carência de estudos que considerem as mudanças comportamentais do paciente mineiro especificamente após a reestruturação dos fluxos do DataSUS em 2020.

Este projeto alinha-se aos estudos identificados ao utilizar a base oficial do DataSUS (SIA/SUS), mas inova ao focar no estado de Minas Gerais com um recorte temporal atualizado (pós-2020) e ao propor o Overbooking Inteligente como ferramenta de intervenção direta, preenchendo a lacuna entre a análise de dados e a eficiência operacional da ponta do atendimento.

# Descrição do _dataset_ selecionado

Nesta seção, é detalhado o conjunto de dados que servirá de base para o modelo de Overbooking Inteligente, focado na eficiência ambulatorial em Minas Gerais.

## Identificação e origem
Nome: Produção Ambulatorial do SUS (SIA/SUS) - Minas Gerais.

Fonte: Ministério da Saúde / DATASUS - portal TABNET.

Link de acesso: TABNET - Produção Ambulatorial

Licença de uso: Dados abertos de uso público (Lei de Acesso à Informação).

## Visão geral
Total de registros: Estimado em mais de 1.500 linhas de dados agregados (cruzamento por macrorregião, complexidade e período).

Período coberto: Janeiro de 2020 a Dezembro de 2025 (projeções baseadas em competências processadas).

Contextualização: O dataset agrupa a produção de saúde em Minas Gerais após o marco da pandemia de COVID-19. Ele não foca no paciente individual (LGPD), mas na eficiência operacional das unidades, comparando o que foi solicitado pelo gestor versus o que foi efetivamente executado.

## Atributos
A tabela abaixo descreve os campos selecionados após o cruzamento multidimensional no portal:
| Nome do Atributo | Descrição | Tipo | Unidade | Exemplos |
| :--- | :--- | :--- | :--- | :--- |
| Macrorregião de Saúde | Região administrativa de saúde em MG | String | Localidade | Centro, Sul |
| Complexidade | Nível tecnológico do procedimento | String | Categoria | Média Complexidade |
| Caráter Atendiment | Natureza da consulta/exame | String | Categoria | Eletivo, Urgência |
| Grupo Procedimento | Tipo de ação de saúde realizada | String | Categoria | 03 Procedimentos Clínicos |
| Mês/Ano | Competência cronológica do dado | Date | Mês/Ano | Jan/2020, Out/2025 |
| Qtd.apresentada | Quantidade de atendimentos agendados | Int | Unidades | 1500, 240, 50 |
| Qtd.aprovada | Quantidade de atendimentos realizados | Int | Unidades | 1200, 200, 45 |

## Qualidade dos dados
- Valores Faltantes: Baixa incidência, pois são dados de faturamento obrigatório. Registros rotulados como "Ignorado" serão descartados no pré-processamento.
- Inconsistências: Pode haver divergência temporal entre a data do atendimento e o mês de processamento (competência). O grupo utilizará a data de competência como referência padrão.
- Duplicatas: Por ser uma base agregada pelo TABNET, não há risco de duplicatas de registros individuais, apenas linhas repetidas caso a extração seja feita por múltiplos filtros sobrepostos.
- Outliers: Espera-se encontrar quedas bruscas (anomalias) na Qtd.aprovada durante os picos da pandemia (2020-2021). Esses dados serão tratados para não enviesar o modelo de overbooking em períodos de normalidade.

# Canvas analítico

Nesta seção, você deverá estruturar e preencher o seu Canvas Analítico, que tem como objetivo registrar a organização das ideias e apresentar o modelo de negócio do projeto.

O Canvas deve ser preenchido integralmente, mesmo que algumas informações ainda não estejam totalmente definidas. Nessa etapa inicial, é aceitável trabalhar com hipóteses ou estimativas, desde que sejam coerentes com o problema e o contexto definidos.

**Dica:** O Canvas Analítico serve como guia visual para alinhar expectativas e direcionar o desenvolvimento. Ele poderá (e deverá) ser revisitado e atualizado ao longo do projeto.

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

- Integrante 1: Introdução e Problema
- Integrante 2: Questão de Pesquisa e Objetivos
- Integrante 3: Justificativa e Público-Alvo
- Integrante 4: Estado da Arte
- Integrante 5: Dataset e Extração
- Integrante 6: Canvas Analítico e Solução

# Referências

BRASIL. Ministério da Saúde. Departamento de Informática do SUS (DATASUS). **Sistema de Informações Ambulatoriais do SUS (SIA/SUS)**. Brasília, DF: Ministério da Saúde, 2026. Disponível em: https://datasus.saude.gov.br/transferencia-de-arquivos/. Acesso em: 6 mar. 2026.

BRASIL. Ministério da Saúde. Departamento de Informática do SUS (DATASUS). **TABNET: Produção Ambulatorial do SUS - Minas Gerais**. Brasília, DF: Ministério da Saúde, 2026. Disponível em: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sia/cnv/pamm.def. Acesso em: 6 mar. 2026.

MINAS GERAIS. Secretaria de Estado de Saúde. **Relatório de Eficiência dos Consórcios Intermunicipais de Saúde: 2020-2024**. Belo Horizonte: SES-MG, 2024.

OLIVEIRA, M.; LIMA, R. Otimização de agendas médicas ambulatoriais via overbooking probabilístico e simulação de Monte Carlo. **Revista Brasileira de Informática em Saúde**, São Paulo, v. 15, n. 1, p. 112-128, 2022.

REIS, F. et al. Uso de modelos de linguagem (BERTimbau) para classificação de motivos de absenteísmo no SUS. **Journal of Health Informatics**, São Paulo, v. 16, n. 3, p. 89-102, set. 2024.

SANTOS, L. R. et al. Impacto da pandemia de COVID-19 no absenteísmo em consultas especializadas em Minas Gerais. **Revista de Saúde Pública de Minas Gerais**, Belo Horizonte, v. 14, n. 2, p. 45-60, 2022.

SILVA, A. C. O absenteísmo pós-pandemia: uma análise preditiva na região Sudeste via SIA/SUS. **Cadernos de Saúde Pública**, Rio de Janeiro, v. 39, n. 4, e001234, 2023.
