# Aplicação web — Etapa 5

## Objetivo

Protótipo acadêmico para inferência dinâmica de probabilidade de não comparecimento (*no-show*) e simulação de política conservadora de overbooking, sem banco de dados e sem persistência de formulários.

## Arquitetura

- **Backend:** FastAPI + Pydantic + scikit-learn/joblib
- **Frontend:** Jinja2 + HTML semântico + CSS/JS próprios
- **Modelo:** artefato pré-treinado em `artifacts/no_show_model.joblib`
- **Deploy:** Vercel com Root Directory `src/webapp`

## Estrutura

```text
src/webapp/
├── app.py
├── ml/
├── templates/
├── public/
├── artifacts/
├── scripts/
└── tests/
```

## Instalação

```bash
cd src/webapp
python -m pip install -r requirements.txt
```

Para treinar/reexportar o modelo:

```bash
python -m pip install -r requirements-train.txt
python scripts/train_export_model.py
python scripts/validate_artifact.py
```

## Execução local

```bash
cd src/webapp
uvicorn app:app --reload
```

Acesse `http://127.0.0.1:8000`.

## Testes

```bash
cd src/webapp
python scripts/validate_artifact.py
pytest -q
```

## Deploy no Vercel

1. Criar projeto no Vercel apontando para este repositório.
2. Definir **Root Directory:** `src/webapp`
3. Framework Preset: **Other**
4. Build Command: vazio (ou padrão do runtime Python)
5. Output Directory: padrão
6. Garantir que `artifacts/no_show_model.joblib` e `artifacts/model_metadata.json` estejam versionados.

Não é necessário variável de ambiente para inferência básica.

## Privacidade

- Sem banco de dados
- Sem autenticação
- Sem cookies de rastreamento
- Sem analytics
- Dados do formulário não são armazenados nem logados

## Limitações

- Dataset de Vitória/ES e período histórico específico
- Probabilidades são estimativas, não decisões automáticas
- Divergência documentada entre texto acadêmico (XGBoost campeão) e critério de deploy (Random Forest em empate técnico de PR-AUC)
