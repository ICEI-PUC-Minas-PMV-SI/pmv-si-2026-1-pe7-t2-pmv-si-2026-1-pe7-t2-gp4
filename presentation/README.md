# Apresentação da Solução

Este projeto foi estruturado seguindo as melhores práticas da Ciência de Dados, evoluindo desde a compreensão bruta dos dados até a entrega de uma solução preditiva integrada à operação de negócio. Abaixo está a jornada consolidada do nosso desenvolvimento:

### Etapa 1 & 2: Exploração de Dados (EDA) e Alinhamento de Negócio
* **O Problema:** O absenteísmo (~20,2% de *no-show*) gerava um gargalo crítico na eficiência do SUS em Vitória (ES).
* **Os Dados:** Analisamos mais de 110 mil registros e identificamos que o principal causador de faltas era o fator temporal: a variável `DiasEspera` (janela de tempo entre a marcação e o dia da consulta) concentrava a maior força preditiva do fenômeno.
* **Saneamento:** Realizamos a limpeza de inconsistências (como idades negativas) e preparamos as variáveis categóricas sem inflar artificialmente a dimensionalidade dos dados.

### Etapa 3: Governança Estatística e Baseline contra Vazamento
* **Combate ao Data Leakage:** Implementamos uma divisão rigorosa baseada em Grupos (`GroupShuffleSplit` por `PatientId`). Isso garantiu que o histórico de um mesmo paciente nunca estivesse simultaneamente no treino e no teste, blindando o modelo contra métricas superestimadas e irreais.
* **Primeiro Avanço:** Estabelecemos o modelo *incumbente* com Random Forest, que serviu como ponto de partida (*baseline*) robusto para o mapeamento de interações não-lineares, superando abordagens lineares simples.

### Etapa 4: Estado da Arte, Métricas Reais e Validação Operacional
* **Modelagem de Elite:** Introduzimos os algoritmos de *Gradient Boosting* (**XGBoost** e **LightGBM**). Identificamos que o projeto atingiu o "teto de performance" das *features* disponíveis (~0,35 de PR-AUC), provando que melhorias futuras dependem de novos dados, e não de modelos mais complexos. O XGBoost foi o campeão pela excelente calibração de suas probabilidades.
* **Foco no Negócio (PR-AUC):** Substituímos métricas frágeis por análises focadas na classe de interesse (a falta). Demonstramos por meio de *Threshold Tuning* que fixar um limiar binário rígido (ex: 0,50 ou 0,70) destrói o valor do modelo para a tomada de decisão em saúde.
* **O Fechamento Operacional:** Desenvolvemos uma **Política de Overbooking Inteligente** baseada na soma de variáveis de Bernoulli (probabilidade pura). O pipeline simulado na base de auditoria final provou que a unidade de saúde consegue **recuperar 67,0% de sua capacidade ociosa** (1.442 consultas salvas), limitando o risco de superlotação a uma média perfeitamente gerenciável de apenas **1,73 paciente excedente** por dia.

---

# Apresentação Final do Projeto

Assista ao vídeo completo com a defesa da nossa solução, detalhando a metodologia aplicada, a arquitetura do pipeline modular e a tradução dos resultados estatísticos em retorno financeiro e social para a gestão da saúde pública:

**[Clique aqui para assistir à Apresentação Final no YouTube](https://youtu.be/ztCxngAZkRU)**

