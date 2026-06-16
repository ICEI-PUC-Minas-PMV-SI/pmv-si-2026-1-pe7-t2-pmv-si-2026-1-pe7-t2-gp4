# Relatório Técnico — Etapa 5 (Aplicação Web)

## 1. Objetivo da Etapa 5

Implantar uma aplicação em nuvem capaz de realizar inferência dinâmica com modelos previamente treinados, recebendo novas instâncias via interface web, sem reprocessar o treinamento em tempo de execução.

## 2. Requisito atendido

- Modelo treinado e exportado em `src/webapp/artifacts/`
- API e interface para consulta individual e em lote
- Política de overbooking calculada a partir de probabilidades
- Nenhum dado preenchido é persistido
- Sem banco de dados

## 3. Arquitetura

Monólito Python com FastAPI servindo API REST e páginas Jinja2, arquivos estáticos em `public/` e módulo `ml/` para validação, pré-processamento, inferência e política operacional.

## 4. Tecnologias

Python 3.11+, FastAPI, Pydantic, Jinja2, scikit-learn, joblib, NumPy, pandas (treinamento), pytest, Vercel Python Runtime.

## 5. Modelo implantado

- **Modelo:** Random Forest (`RandomForestClassifier`)
- **Motivo:** PR-AUC principal com empate técnico em relação ao XGBoost (diferença ≤ 0,005). Em empate, priorizou-se menor dependência de produção e facilidade de deploy.
- **Features:** Age, DiasEspera, Gender_numeric, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received
- **Métricas reais (teste, reprodução 16/06/2026):**
  - PR-AUC: 0,3564
  - ROC-AUC: 0,7264
  - Precisão (0,50): 0,3108
  - Recall (0,50): 0,8111
  - F1 (0,50): 0,4494
  - Acurácia: 0,5938
- **Registros:** 110.527 originais → 110.521 após limpeza
- **Taxa histórica de faltas:** 20,19%

## 6. Pré-processamento

- Datas normalizadas com `dt.normalize()` antes de `DiasEspera`
- Remoção de `DiasEspera < 0` (5 registros) e `Age < 0` (1 registro)
- Gênero: F=0, M=1
- Alvo: compareceu=0, faltou=1
- Split: `GroupShuffleSplit` por `PatientId`, 70/30, `random_state=42`

## 7. Consulta dinâmica

Endpoints:

- `POST /api/predict` — probabilidade individual
- `POST /api/predict-batch` — agenda com até 100 pacientes
- `GET /api/health` — status do artefato

## 8. Simulador individual

Formulário com validação backend, exemplo preenchível, faixas de risco comunicacionais (baixo/moderado/alto) e avisos éticos.

## 9. Simulador de agenda

Tabela editável, cálculo em lote, soma de probabilidades, faltas esperadas, fator de segurança, teto e recomendação final.

## 10. Política de overbooking

```text
expected_absences = soma(probabilidades)
preliminary_slots = floor(expected_absences × safety_factor)
hard_cap = floor(pacientes × max_extra_percentage)
recommended_extra_slots = min(preliminary_slots, hard_cap)
```

Padrão: `safety_factor=0,85`, `max_extra_percentage=0,15`.

## 11. Testes

Executados em `src/webapp`:

```bash
python scripts/validate_artifact.py
pytest -q
```

Resultado: **25 testes aprovados**.

## 12. Privacidade e LGPD

Sem identificadores pessoais, sem persistência, sem analytics, sem cookies de rastreamento. Uso acadêmico com decisão humana obrigatória.

## 13. Capacidade

Limite de 100 pacientes por requisição em lote. Modelo carregado uma única vez na memória do processo.

## 14. Deploy

- **Root Directory Vercel:** `src/webapp`
- **Arquivo:** `vercel.json`
- **Comando local:** `uvicorn app:app --reload`
- Deploy de produção: pendente de autorização explícita da equipe.

## 15. Limitações

- Generalização limitada ao contexto do dataset
- Associações estatísticas, não causalidade
- Métricas variam levemente conforme versões de bibliotecas

## 16. Divergências encontradas

1. **Modelo campeão:** documentação da Etapa 4 declara XGBoost campeão; reprodução isolada registrou PR-AUC 0,3579 (XGB) vs 0,3564 (RF), empate técnico — implantado RF.
2. **Split:** documentação da Etapa 4 cita 77.431/33.091; reprodução obteve 77.565/32.956 (mesma estratégia, possível variação de versão).
3. **Métricas:** valores do notebook (ex.: PR-AUC 0,3512 para RF) diferem na 2ª–3ª casa decimal da reprodução atual, mantendo o mesmo ranqueamento qualitativo.

## 17. Conclusão

A Etapa 5 foi implementada de forma isolada em `src/webapp/`, preservando notebooks e relatórios anteriores, com aplicação funcional pronta para demonstração e deploy no Vercel.
