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

### 1. Framework de Análise de Decisão para No-Shows (Deina, 2024)
Problema e contexto: O estudo abordou a previsão de não comparecimentos a consultas médicas ambulatoriais para mitigar os impactos das faltas no sistema, como o desperdício de recursos e a necessidade de estratégias de overbooking.

Dados (dataset): Utilizou bancos de dados estruturados e desbalanceados da área médica, focando na identificação de padrões de comportamento de pacientes.

Abordagem/algoritmos: Explorou o uso de algoritmos avançados, especificamente a Symbolic Regression (SR) como modelo de previsão e a técnica Instance Hardness Threshold (IHT) como estratégia de balanceamento de dados.

Métricas de avaliação: O foco foi garantir a generalização dos resultados em situações do mundo real e reduzir o viés nos conjuntos de calibração e validação.

Resultados: A aplicação da técnica IHT melhorou a precisão na previsão dos no-shows, fortalecendo a interpretabilidade dos resultados para apoio à decisão médica.

### 2. Machine Learning para No-Show no Setor de Saúde (Salazar et al., 2022)
   
Problema e contexto: Aplicação de técnicas de aprendizado de máquina para prever o não comparecimento de pacientes, visando otimizar a eficiência e reduzir perdas financeiras em instituições de saúde.

Dados (dataset): Dataset real coletado no Centro Especializado em Reabilitação (CER II) da Univali, Santa Catarina. As variáveis incluíram códigos CID-10, dados climáticos históricos do INMET e dados regionais da AMFRI.

Abordagem/algoritmos: Implementação de modelos utilizando as bibliotecas Python Pandas e NumPy para processamento e classificação.

Métricas de avaliação: Validação do modelo preditivo para a realidade local da instituição.

Resultados: O estudo demonstrou a viabilidade de prever as faltas integrando dados de saúde com variáveis externas (como o clima), auxiliando na gestão hospitalar.

### 3. Modelo Preditivo e Avaliação Clínica (Valero-Bover et al., 2022)
   
Problema e contexto: Desenvolvimento e validação clínica de um modelo preditivo para reduzir a não frequência em consultas ambulatoriais.

Dados (dataset): Registros de consultas ambulatoriais focados no desenvolvimento de um workflow robusto.

Abordagem/algoritmos: Modelagem baseada em Machine Learning integrada a um processo de avaliação clínica.

Métricas de avaliação: Acurácia preditiva e impacto na redução efetiva do absenteísmo após a intervenção clínica.

Resultados: O trabalho destacou que a predição deve estar aliada a uma validação clínica rigorosa para ser eficaz no ambiente hospitalar.

### 4. Análise de Variáveis em Hospital Terciário (Baptista et al., 2023)
   
Problema e contexto: Estudo transversal para analisar variáveis relacionadas ao agendamento de consultas que resultaram em no-show em ambulatórios de um hospital de alta complexidade.

Dados (dataset): Base de dados de um hospital terciário, analisando o comportamento dos pacientes agendados.

Abordagem/algoritmos: Análise estatística transversal das causas de ausência.

Métricas de avaliação: Frequência de recebimento de mensagens de lembrete e tempo de espera entre agendamento e consulta.

Resultados: Identificou-se que 49,8% dos pacientes não receberam lembretes e que tempos de espera longos (180 a 365 dias para 36,6% dos casos) são fatores críticos que aumentam o no-show.

### 5. Estratégia "Faltômetro" na Atenção Básica (Silva et al., 2021)
Problema e contexto: Enfrentamento do absenteísmo recorrente na Atenção Básica através de uma ferramenta de monitoramento e diálogo.

Dados (dataset): Relato de experiência coletando dados de agendamentos em unidades de saúde primária.

Abordagem/algoritmos: Criação do "Faltômetro", uma ferramenta visual e pedagógica para consolidar as faltas e promover debates em salas de espera.

Métricas de avaliação: Impacto qualitativo na percepção de profissionais e usuários sobre as causas das faltas.

Resultados: A estratégia foi eficaz para reduzir a culpabilização do usuário e permitiu entender que o absenteísmo na atenção básica requer um diálogo horizontal e educação em saúde para ser mitigado.

## Texto-síntese Crítico
Os estudos recentes concordam que o absenteísmo é um problema multifatorial que não depende apenas da vontade do paciente, mas de falhas nos processos de comunicação (como a falta de lembretes) e barreiras estruturais, como longos tempos de espera. Tecnicamente, há um consenso de que lidar com dados desbalanceados é o maior desafio para a IA nesta área, exigindo técnicas específicas como IHT ou Symbolic Regression para garantir que os modelos não sejam tendenciosos e consigam de fato identificar quem irá faltar.
Divergências surgem no foco da solução: enquanto trabalhos como os de Salazar (2022) e Deina (2024) buscam a sofisticação algorítmica para prever o comportamento, abordagens como o "Faltômetro" (2021) focam na humanização e na reorganização institucional como forma de resolver a raiz do problema. Permanece a lacuna de integrar essas duas pontas: modelos de IA que não apenas prevejam a falta, mas que sugiram intervenções personalizadas (socioeconômicas ou clínicas) integradas ao workflow dos profissionais de saúde.
Seu projeto se alinha a esta tendência ao buscar identificar no-shows com IA, posicionando-se na fronteira tecnológica de aprimorar a tomada de decisão. A literatura atual sugere que o sucesso de sua solução dependerá da capacidade de equilibrar a robustez preditiva com a interpretabilidade, permitindo que gestores compreendam por que o paciente está faltando para agir preventivamente de forma eficaz.

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
Para guiar visualmente alinhando expectativas e direcionando o desenvolvimento do projeto, segue abaixo o Canvas Analítico. 
Ele poderá (e deverá) ser revisitado e atualizado ao longo do projeto, já que como o projeto se encontra em uma etapa inicial, é aceitável trabalhar com hipóteses ou estimativas, desde que sejam coerentes com o problema e o contexto definidos.

<img width="1084" height="767" alt="image" src="https://github.com/user-attachments/assets/e8965fe3-e772-44d1-8b39-5d49469e8042" />

# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

# Referências

BRASIL. Ministério da Saúde. Departamento de Informática do SUS (DATASUS). **Sistema de Informações Ambulatoriais do SUS (SIA/SUS)**. Brasília, DF: Ministério da Saúde, 2026. Disponível em: https://datasus.saude.gov.br/transferencia-de-arquivos/. Acesso em: 6 mar. 2026.

BRASIL. Ministério da Saúde. Departamento de Informática do SUS (DATASUS). **TABNET: Produção Ambulatorial do SUS - Minas Gerais**. Brasília, DF: Ministério da Saúde, 2026. Disponível em: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sia/cnv/pamm.def. Acesso em: 6 mar. 2026.

BAPTISTA, S. C. P. D. et al. Estudo transversal sobre ausências de pacientes em consultas médicas agendadas em ambulatórios de hospital terciário. Enfermagem em Foco, v. 14, 2023.

CESÁRIO, I. R. A redução do absenteísmo de usuários em consultas de especialidades no SUS: a experiência de um instituto de referência no Rio de Janeiro. Dissertação, 2022.

DEINA, C. Aprimorando a Tomada de Decisão em Saúde com Aprendizado de Máquina em Problemas de Classificação em Dados Desbalanceados. Tese (Doutorado em Engenharia de Produção) – UFRGS, Porto Alegre, 2024.

SALAZAR, L. H. A. et al. Application of Machine Learning Techniques to Predict a Patient’s No-Show in the Healthcare Sector. Future Internet, v. 14, n. 3, 2022.

SILVA, M. T. A. et al. “Faltômetro”: Estratégia para o enfrentamento do absenteísmo no âmbito da Atenção Básica. Revista Ciência Plural, v. 7, n. 2, p. 163–176, 2021.

VALERO-BOVER, D. et al. Reducing non-attendance in outpatient appointments: predictive model development, validation, and clinical assessment. BMC Health Services Research, v. 22, n. 1, 2022.
