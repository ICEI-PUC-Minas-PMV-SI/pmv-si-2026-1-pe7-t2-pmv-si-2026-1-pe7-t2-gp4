# Introdução

Este projeto apresenta uma proposta de otimização para a gestão de agendas no Sistema Único de Saúde (SUS), focando no absenteísmo (*no-show*) em consultas médicas. Utilizando o dataset **"Medical Appointment No Shows" (Kaggle)**, que contém mais de 110 mil registros históricos de consultas da cidade de Vitória (Espírito Santo), o trabalho propõe a aplicação de modelos de Aprendizado de Máquina para prever a probabilidade de falta de pacientes e fundamentar uma estratégia de **overbooking dinâmico e inteligente**. O objetivo é maximizar a ocupação das agendas médicas, garantindo que ociosidades previstas sejam compensadas por agendamentos excedentes de forma segura, acelerando a fila de espera da população.

## Problema

O problema central é a ineficiência causada pelo absenteísmo em consultas agendadas, que gera ociosidade de médicos e desperdício de infraestrutura na rede pública de saúde.

- **Contexto:** O sistema de saúde sofre com altas taxas de não comparecimento. Ao contrário de dados agregados, a previsão de faltas depende do comportamento individual e de barreiras logísticas ou de comunicação.
- **Impacto:** Horários vagos que poderiam ser utilizados por pacientes que aguardam meses em filas de espera para especialidades críticas não são aproveitados.
- **Fatores Agravantes:** A falta de lembretes (SMS) e o longo tempo de espera entre a data de marcação e a data da consulta são apontados como os maiores causadores da evasão.

## Questão de pesquisa

Qual a probabilidade de um paciente faltar a uma consulta agendada, com base em seu perfil e histórico, e como podemos utilizar essa predição de modelos de *Machine Learning* para aplicar uma margem de *overbooking* seguro nas unidades de saúde?

## Objetivos preliminares

O objetivo geral é experimentar modelos de aprendizado de máquina adequados para prever o risco individual de *no-show* e propor um sistema de recomendação de vagas excedentes (overbooking) para as recepções das unidades.

### Objetivos específicos:

- Realizar a Análise Exploratória e *Feature Engineering* na base de dados (ex: calcular a diferença em dias entre a marcação e a consulta);
- Comparar o desempenho de algoritmos de classificação (como Random Forest e Regressão Logística) focando em alta precisão para evitar falsos positivos;
- Desenvolver a "lógica de negócio" do overbooking, que calculará o limite de pacientes extras que podem ser chamados por dia sem causar superlotação na sala de espera.

## Justificativa

A escolha do dataset do Kaggle justifica-se pela granularidade dos dados. Diferente de bases de faturamento, essa base contém informações a nível do **paciente** (idade, comorbidades, recebimento de SMS, tempo de espera).

- **Relevância:** Lidar com dados comportamentais permite que a solução não apenas preveja a falta, mas ajude o gestor a entender o *motivo* da falta.
- **Impacto:** Transformar a predição passiva em uma ação corretiva ativa (o *overbooking*). Zerar a ociosidade médica diminui o custo por atendimento e aumenta drasticamente a produtividade da atenção básica e especializada.

## Público-Alvo

- **Secretarias Municipais de Saúde e Coordenadores de UBS:** Decisores que precisam otimizar o uso do tempo médico e aumentar a capacidade de atendimento sem contratar mais profissionais.
- **Recepcionistas e Atendentes de Regulação:** Usuários finais que utilizarão o painel/dashboard para saber exatamente quantos pacientes a mais podem agendar na grade do dia seguinte.

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

Nesta seção, é detalhado o conjunto de dados que servirá de base para o modelo preditivo e para a estratégia de Overbooking Inteligente.

## Identificação e origem
- **Nome:** Medical Appointment No Shows
- **Fonte:** Portal Kaggle (Dados originais da Prefeitura de Vitória, Espírito Santo).
- **Link de acesso:** [Kaggle - Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments/data)
- **Licença de uso:** Dados abertos de uso público / CC0: Public Domain.

## Visão geral
- **Total de registros:** 110.527 linhas (consultas médicas).
- **Contextualização:** O dataset mapeia o agendamento de consultas na rede pública de saúde da capital capixaba. O grande diferencial desta base é focar em características do paciente e da marcação, informando se ele possui condições crônicas, se está em programas sociais, se recebeu aviso por SMS e, por fim, se compareceu ou não à consulta.

## Atributos Principais
A tabela abaixo descreve os campos mais relevantes para a nossa análise:

| Nome do Atributo | Descrição | Tipo | Exemplos |
| :--- | :--- | :--- | :--- |
| **PatientId** | Identificador único do paciente | Float | 2.987250e+13 |
| **ScheduledDay** | Data e hora em que a consulta foi marcada | DateTime | 2016-04-29T18:38:08Z |
| **AppointmentDay** | Data em que a consulta vai acontecer | DateTime | 2016-04-29T00:00:00Z |
| **Age** | Idade do paciente | Int | 62, 18, 5 |
| **Neighbourhood** | Bairro onde a consulta ocorrerá | String | JARDIM DA PENHA |
| **Scholarship** | Indica se o paciente recebe Bolsa Família (0 = Não, 1 = Sim) | Int (Booleano) | 0, 1 |
| **Hipertension / Diabetes** | Indica presença de comorbidades crônicas | Int (Booleano) | 0, 1 |
| **SMS_received** | Indica se 1 ou mais mensagens SMS de lembrete foram enviadas | Int | 0, 1 |
| **No-show** | Variável Alvo (Target). 'No' (Não faltou/Compareceu) ou 'Yes' (Faltou) | String | No, Yes |

## Qualidade dos dados
- **Valores Faltantes (Nulls):** O dataset original é muito limpo e não apresenta valores nulos em suas colunas principais.
- **Inconsistências Temporais:** Algumas datas de marcação (`ScheduledDay`) aparecem no futuro em relação à data da consulta (`AppointmentDay`), o que representa um erro de digitação do sistema e precisará ser filtrado.
- **Outliers:** A coluna idade (`Age`) possui ao menos um registro com valor negativo (-1) e idades extremamente avançadas (acima de 110 anos), que deverão ser tratados no pré-processamento.
- **Feature Engineering:** Será necessário criar uma nova variável calculando a diferença em dias entre `ScheduledDay` e `AppointmentDay`, pois a literatura indica que esta é a variável preditora mais forte para o *no-show*.

# Canvas analítico
Para guiar visualmente, alinhando expectativas e direcionando o desenvolvimento do projeto, segue abaixo o Canvas Analítico. 
Ele poderá (e deverá) ser revisitado e atualizado ao longo do projeto, já que como o projeto se encontra em uma etapa inicial, é aceitável trabalhar com hipóteses ou estimativas, desde que sejam coerentes com o problema e o contexto definidos.

<img width="1084" height="767" alt="image" src="https://github.com/user-attachments/assets/e8965fe3-e772-44d1-8b39-5d49469e8042" />

# Vídeo de apresentação da Etapa 01



# Referências

BAPTISTA, S. C. P. D. et al. Estudo transversal sobre ausências de pacientes em consultas médicas agendadas em ambulatórios de hospital terciário. Enfermagem em Foco, v. 14, 2023.

DEINA, C. Aprimorando a Tomada de Decisão em Saúde com Aprendizado de Máquina em Problemas de Classificação em Dados Desbalanceados. Tese (Doutorado em Engenharia de Produção) – UFRGS, Porto Alegre, 2024.

KAGGLE. **Medical Appointment No Shows**. Disponível em: https://www.kaggle.com/datasets/joniarroba/noshowappointments/data. Acesso em: mar. 2026.

SALAZAR, L. H. A. et al. Application of Machine Learning Techniques to Predict a Patient’s No-Show in the Healthcare Sector. Future Internet, v. 14, n. 3, 2022.

SILVA, M. T. A. et al. “Faltômetro”: Estratégia para o enfrentamento do absenteísmo no âmbito da Atenção Básica. Revista Ciência Plural, v. 7, n. 2, p. 163–176, 2021.

VALERO-BOVER, D. et al. Reducing non-attendance in outpatient appointments: predictive model development, validation, and clinical assessment. BMC Health Services Research, v. 22, n. 1, 2022.
