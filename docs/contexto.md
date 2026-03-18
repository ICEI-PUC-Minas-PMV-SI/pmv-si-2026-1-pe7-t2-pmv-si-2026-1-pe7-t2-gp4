# Introdução

Este projeto apresenta uma proposta de otimização para a gestão de agendas no Sistema Único de Saúde (SUS), focando no absenteísmo (*no-show*) em consultas médicas. Utilizando o dataset **"Medical Appointment No Shows" (Kaggle)**, que contém mais de 110 mil registros históricos de consultas da cidade de Vitória (Espírito Santo), o trabalho propõe a aplicação de modelos de Aprendizado de Máquina para prever a probabilidade de falta de pacientes e fundamentar uma estratégia de **overbooking dinâmico e inteligente**. O objetivo é maximizar a ocupação das agendas médicas, garantindo que ociosidades previstas sejam compensadas por agendamentos excedentes de forma segura, acelerando a fila de espera da população.

## Problema

O problema central é a ineficiência causada pelo absenteísmo em consultas agendadas, que gera ociosidade de médicos e desperdício de infraestrutura na rede pública de saúde.

- **Contexto:** O sistema de saúde sofre com altas taxas de não comparecimento. Ao contrário de dados agregados, a previsão de faltas depende do comportamento individual e de barreiras logísticas ou de comunicação.
- **Impacto:** Horários vagos que poderiam ser utilizados por pacientes que aguardam meses em filas de espera para especialidades críticas não são aproveitados.
- **Fatores Agravantes:** A falta de lembretes (SMS) e o longo tempo de espera entre a data de marcação e a data da consulta são apontados como os maiores causadores da evasão.

## Questão de pesquisa

Como prever o risco de no-show e utilizá-lo para definir um overbooking seguro?

## Objetivos preliminares

O objetivo geral é experimentar modelos de aprendizado de máquina adequados para prever o risco individual de *no-show* e propor um sistema de recomendação de vagas excedentes (overbooking) para as recepções das unidades.

### Objetivos específicos:

- Desenvolver um modelo preditivo de risco de absenteísmo utilizando técnicas de Aprendizado de Máquina, capaz de estimar a probabilidade individual de não comparecimento às consultas;
- Construir um conjunto de variáveis derivadas (features) relevantes, incorporando características clínicas, sociais e comportamentais dos pacientes para aprimorar o desempenho do modelo;
- Implementar um mecanismo de overbooking inteligente, baseado na previsão de no-show, permitindo definir limites seguros de agendamentos excedentes para cada dia ou especialidade;
- Criar um protótipo de sistema ou dashboard operacional (frontend) para apoiar recepcionistas e gestores na visualização das estimativas de absenteísmo e das recomendações de vagas adicionais;
- Avaliar o impacto potencial do overbooking baseado em IA na ocupação das agendas médicas, comparando cenários com e sem a aplicação do modelo.

## Justificativa

O absenteísmo em consultas médicas representa um desafio significativo para o Sistema Único de Saúde, pois gera ociosidade profissional, desperdício de recursos e aumento das filas de espera. A ausência de mecanismos preditivos faz com que a gestão das agendas seja predominantemente reativa, resultando em baixa eficiência operacional.

Diante desse cenário, justifica-se o desenvolvimento de uma solução baseada em Aprendizado de Máquina capaz de estimar o risco individual de não comparecimento e subsidiar estratégias de overbooking seguro. Essa abordagem permite otimizar a ocupação das agendas, ampliar o acesso aos serviços de saúde e promover o uso racional dos recursos públicos. Assim, o projeto demonstra relevância social e técnica ao alinhar inovação tecnológica com necessidades reais da gestão em saúde.

## Público-Alvo

- **Secretarias Municipais de Saúde e Coordenadores de UBS:** Decisores que precisam otimizar o uso do tempo médico e aumentar a capacidade de atendimento sem contratar mais profissionais.
- **Recepcionistas e Atendentes de Regulação:** Usuários finais que utilizarão o painel/dashboard para saber exatamente quantos pacientes a mais podem agendar na grade do dia seguinte.

## Estado da arte

Diversos estudos têm investigado o absenteísmo em consultas médicas na saúde pública brasileira e internacional, buscando compreender suas causas e desenvolver estratégias para reduzir seus impactos. No contexto nacional, um dos trabalhos mais recentes é o de Deina (2024), que propõe um framework de apoio à decisão baseado em Aprendizado de Máquina para lidar com bases desbalanceadas no setor de saúde. A autora utiliza técnicas como Symbolic Regression e Instance Hardness Threshold para aprimorar a capacidade dos modelos de identificar pacientes com maior probabilidade de faltar, contribuindo especialmente para a etapa de modelagem preditiva. Embora se aproxime do presente projeto ao tratar diretamente do desafio do no-show, o estudo concentra-se na robustez estatística dos algoritmos e não avança para aplicações operacionais, como a otimização de agendas.

Em um estudo aplicado ao contexto brasileiro, Salazar et al. (2022) exploram técnicas de Machine Learning para prever o não comparecimento em um centro de reabilitação, utilizando variáveis clínicas, demográficas e ambientais. A pesquisa comprova a viabilidade de integrar dados heterogêneos para melhorar a acurácia preditiva e reforça a importância de considerar múltiplos fatores no comportamento de presença do paciente. Assim como no presente projeto, a proposta utiliza modelos de classificação para antecipar o absenteísmo; entretanto, o estudo não aborda como essa previsão pode ser incorporada a mecanismos de gestão, como o uso de overbooking inteligente para preencher vagas que seriam desperdiçadas.

Em um cenário internacional, Valero-Bover et al. (2022) desenvolvem um modelo preditivo validado clinicamente para consultas ambulatoriais, destacando que a eficácia de soluções dessa natureza depende não apenas da performance algorítmica, mas também da aceitação por parte dos profissionais de saúde. Essa perspectiva se relaciona ao presente trabalho ao demonstrar que abordagens tecnológicas devem estar integradas ao fluxo operacional das unidades. Contudo, os autores se concentram na construção e validação clínica do modelo, sem discutir estratégias de utilização prática das previsões para otimização de agendas.
Outra contribuição relevante para o entendimento do fenômeno do absenteísmo é apresentada por Baptista et al. (2023), que analisam, de forma transversal, as variáveis relacionadas às faltas em um hospital de alta complexidade. Os autores identificam que a ausência de lembretes e os longos tempos de espera são fatores críticos que aumentam significativamente a probabilidade de no-show. Embora o estudo não utilize técnicas de Machine Learning, ele fornece evidências sobre variáveis essenciais que devem ser consideradas na construção de modelos preditivos, reforçando a necessidade de incorporar dimensões comportamentais e de comunicação com o paciente.

Por outro lado, Silva et al. (2021) abordam o absenteísmo a partir de uma perspectiva mais humana e educativa por meio da criação do “Faltômetro”, ferramenta que visa promover diálogo entre profissionais e usuários a partir da visualização das faltas. Embora não envolva técnicas analíticas avançadas, essa abordagem evidencia que o absenteísmo não pode ser compreendido apenas como um evento estatístico, mas deve considerar fatores sociais, comportamentais e estruturais que influenciam a presença do paciente. A diferença fundamental em relação ao presente projeto está no enfoque: enquanto Silva et al. enfatizam mudanças culturais e comunicativas nas unidades, esta pesquisa propõe uma solução tecnológica voltada para a gestão operacional.

Ao relacionar os estudos revisados ao presente trabalho, observa-se que, embora diferentes em abordagem, todos convergem para a compreensão de que o absenteísmo é um problema multifatorial que exige soluções integradas. As pesquisas baseadas em Machine Learning (Deina, 2024; Salazar et al., 2022; Valero‑Bover et al., 2022) oferecem avanços importantes na predição das ausências, mas não apresentam mecanismos para utilização prática das previsões no planejamento das agendas. Já os estudos focados em análises de fatores e intervenções educativas (Baptista et al., 2023; Silva et al., 2021) contribuem para identificar elementos críticos do comportamento do paciente, mas deixam de lado a automação e o apoio à decisão. Assim, o presente projeto se diferencia ao integrar elementos de ambas as abordagens, propondo um sistema que utiliza predições de no-show para fundamentar uma estratégia de overbooking seguro, preenchendo uma lacuna na literatura ao transformar a previsão em ação operacional concreta dentro da rotina das unidades de saúde.

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
- **Total de colunas (atributos):** 14 variáveis, conforme documentação oficial do dataset.
- **Contextualização:** O dataset mapeia o agendamento de consultas na rede pública de saúde da capital capixaba. O grande diferencial desta base é focar em características do paciente e da marcação, informando se ele possui condições crônicas, se está em programas sociais, se recebeu aviso por SMS e, por fim, se compareceu ou não à consulta.

## Atributos Principais
A tabela abaixo descreve apenas os campos mais relevantes para esta pesquisa, considerando que o dataset completo possui 14 atributos no total:

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

https://github.com/user-attachments/assets/5b4ee673-05fd-4804-ae8a-b4e0c0768aae

Link para visualização no Canva: https://www.canva.com/design/DAHDYPWw9Rs/lucNj4ItcmS5YbRZxTQaqA/watch?utm_content=DAHDYPWw9Rs&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h8fae076eae 

# Referências
**DEINA, Carolina.** *Aprimorando a tomada de decisão em saúde com aprendizado de máquina em problemas de classificação em dados desbalanceados.* 2024. Tese (Doutorado em Engenharia de Produção) — Universidade Federal do Rio Grande do Sul, Porto Alegre, 2024.
Disponível em: https://lume.ufrgs.br/handle/10183/289543.  
Acesso em: 18 mar. 2026.

**SALAZAR, Luiz Henrique A. et al.** *Application of Machine Learning Techniques to Predict a Patient’s No-Show in the Healthcare Sector.* Future Internet, v. 14, n. 1, p. 3, 2022.  
DOI: 10.3390/fi14010003.  
Disponível em: https://www.mdpi.com/1999-5903/14/1/3.  
Acesso em: 18 mar. 2026.

**VALERO-BOVER, Damià et al.** *Reducing non-attendance in outpatient appointments: predictive model development, validation, and clinical assessment.* BMC Health Services Research, v. 22, 451, 2022.  
DOI: 10.1186/s12913-022-07865-y.  
Disponível em: https://link.springer.com/article/10.1186/s12913-022-07865-y.  
Acesso em: 18 mar. 2026.

**BAPTISTA, Simone Cristina Paixão Dias et al.** *Estudo transversal sobre ausências de pacientes em consultas médicas agendadas em ambulatórios de hospital terciário.* Enfermagem em Foco, v. 14, 2023.  
DOI: 10.21675/2357-707X.2023.v14.e-202346.  
Disponível em: https://enfermfoco.org/wp-content/uploads/articles_xml/2357-707X-enfoco-14-e-202346/2357-707X-enfoco-14-e-202346.pdf.  
Acesso em: 18 mar. 2026.

**SILVA, Maria Tatiane Alves da et al.** *“Faltômetro”: estratégia para o enfrentamento do absenteísmo no âmbito da Atenção Básica.* Revista Ciência Plural, v. 7, n. 2, p. 163–176, 2021.  
DOI: 10.21680/2446-7286.2021v7n2ID22255.  
Disponível em: https://periodicos.ufrn.br/rcp/article/view/22255.  
Acesso em: 18 mar. 2026.

KAGGLE. **Medical Appointment No Shows**. Disponível em: https://www.kaggle.com/datasets/joniarroba/noshowappointments/data. Acesso em: mar. 2026.
