# Eficiência do Projeto

`CURSO: Sistemas de Informação`

`DISCIPLINA: Projeto - Pesquisa e Experimentação em Sistemas de Informação`

`SEMESTRE: 7º`

# Inteligência Operacional contra o Absenteísmo no SUS: Estratégia de Overbooking Responsável

## Introdução Geral

O Sistema Único de Saúde (SUS) enfrenta um desafio crônico que afeta diretamente a sua eficiência e o tempo de espera da população: o alto índice de absenteísmo (*no-show*) em consultas médicas e exames. Quando um paciente falta sem aviso prévio, um recurso público valioso é desperdiçado, a ociosidade dos profissionais de saúde aumenta e a fila de espera se estende desnecessariamente. 

Este projeto propõe uma solução baseada em **Ciência de Dados e Inteligência Artificial** para transformar essa análise passiva em uma ação corretiva e preditiva. Utilizando o *dataset* público *"Medical Appointment No Shows"* (Kaggle), composto por mais de **110 mil registros reais** de agendamentos na cidade de Vitória (ES), o trabalho mapeia padrões comportamentais e temporais — como o tempo de espera entre a marcação e a consulta, a idade do paciente e o impacto de lembretes por SMS — para prever a probabilidade individual de falta.

### Da Predição Estatística ao Impacto de Negócio

O grande diferencial deste projeto é transcender a mera modelagem matemática, fechando a lacuna entre a métrica estatística e a operação prática. A partir das probabilidades de risco geradas por algoritmos avançados de *Gradient Boosting* (XGBoost), implementamos um **mecanismo de otimização de capacidade via Política de Overbooking Inteligente**.

Em vez de adotar uma abordagem linear ou punitiva, o pipeline calcula o valor esperado de faltas agregadas por agenda (unidade × dia) e recomenda, de forma dinâmica e segura, o número ideal de encaixes para cada dia de atendimento. Controlada por alavancas de gestão (fatores de segurança e tetos operacionais rígidos), a política foi simulada em ambiente de auditoria final, demonstrando capacidade de **recuperar até 67,0% da ociosidade das agendas**, mantendo o risco de superlotação (*overflow*) sob estrito controle e totalmente absorvível pela recepção das unidades de saúde.

Dessa forma, o projeto se consolida como uma ferramenta de suporte à decisão (*human-in-the-loop*), viabilizando uma gestão pública eficiente, humanizada e orientada a dados, focada em acelerar o atendimento de quem mais precisa sem sobrecarregar a infraestrutura do SUS.

## Integrantes

- Daniela Sofia Fernandes de Assis
- Gabriel Amorim Santos Maia
- Guilherme Lanza Japolino
- João Gabriel Galdino de Oliveira
- Tales Hein
- Lucas Brandão Guedes

## Orientador

- Neil Paiva Tizzo

---

## Planejamento

| Etapa         | Atividades |
|  :----:   | ----------- |
| ETAPA 1         |[Documentação de Contexto e levantamento dos dados](docs/contexto.md) <br> |
| ETAPA 2         |[Conhecendo os dados](docs/conhecendo-dados.md) <br> |
| ETAPA 3         |[Preparação dos dados, construção e avaliação do modelo proposto](docs/construindo-modelo.md) |
| ETAPA 4         |[Preparação dos dados, construção, avaliação e comparação dos modelos propostos](docs/construcao-de-modelos.md) |
| ETAPA 5         |[Implantação e apresentação da solução](docs/implantação-apresentacao.md) <br>  |

---

## Código

<li><a href="src/README.md"> Código Fonte</a></li>

## Apresentação

<li><a href="presentation/README.md"> Apresentação da solução</a></li>

---

## Aplicação Web e Instruções de Utilização (Etapa 5)

Protótipo FastAPI para previsão dinâmica de no-show e simulação de overbooking responsável.

### 1. Acesso em Produção (Nuvem)
* **URL:** [https://pmv-si-2026-1-pe7-t2-pmv-si-2026-1.vercel.app/](https://pmv-si-2026-1-pe7-t2-pmv-si-2026-1.vercel.app/)
* **Status:** Online / Produção
* **Deploy:** Vercel com Root Directory `src/webapp`

### 2. Execução Local
Para rodar a interface e a API REST localmente na sua máquina, execute os seguintes comandos no terminal:

```bash
cd src/webapp 
pip install -r requirements.txt 
uvicorn app:app --reload
