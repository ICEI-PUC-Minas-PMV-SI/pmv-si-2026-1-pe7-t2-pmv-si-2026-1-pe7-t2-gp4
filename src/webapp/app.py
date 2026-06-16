"""Aplicação web FastAPI — Etapa 5: previsão de no-show e overbooking."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml.inference import build_batch_response, build_prediction_response
from ml.model_loader import load_metadata, model_status
from ml.schemas import BatchPredictRequest, PatientInput

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI(
    title="Previsão de No-Show no SUS",
    description="Protótipo acadêmico de apoio à decisão para overbooking responsável.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if PUBLIC_DIR.is_dir():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")


def _metadata_context() -> dict:
    try:
        return load_metadata()
    except Exception:
        return {}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "interest-cohort=()"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    messages = []
    for error in exc.errors():
        location = " → ".join(str(item) for item in error.get("loc", []))
        message = error.get("msg", "Valor inválido.")
        messages.append(f"{location}: {message}" if location else message)
    return JSONResponse(
        status_code=422,
        content={"detail": messages, "message": "Dados inválidos. Verifique os campos informados."},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if exc.status_code == 404:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "metadata": _metadata_context()},
            status_code=404,
        )
    return templates.TemplateResponse(
        "500.html",
        {"request": request, "metadata": _metadata_context()},
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={"detail": "Erro interno ao processar a solicitação."},
        )
    return templates.TemplateResponse(
        "500.html",
        {"request": request, "metadata": _metadata_context()},
        status_code=500,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "metadata": _metadata_context()},
    )


@app.get("/simulador", response_class=HTMLResponse)
async def simulador(request: Request):
    return templates.TemplateResponse(
        "simulador.html",
        {"request": request, "metadata": _metadata_context()},
    )


@app.get("/agenda", response_class=HTMLResponse)
async def agenda(request: Request):
    return templates.TemplateResponse(
        "agenda.html",
        {"request": request, "metadata": _metadata_context()},
    )


@app.get("/metodologia", response_class=HTMLResponse)
async def metodologia(request: Request):
    return templates.TemplateResponse(
        "metodologia.html",
        {"request": request, "metadata": _metadata_context()},
    )


@app.get("/etica", response_class=HTMLResponse)
async def etica(request: Request):
    return templates.TemplateResponse(
        "etica.html",
        {"request": request, "metadata": _metadata_context()},
    )


@app.get("/sobre", response_class=HTMLResponse)
async def sobre(request: Request):
    return templates.TemplateResponse(
        "sobre.html",
        {"request": request, "metadata": _metadata_context()},
    )


@app.get("/api/health")
async def health():
    return model_status()


@app.post("/api/predict")
async def predict(patient: PatientInput):
    return build_prediction_response(patient)


@app.post("/api/predict-batch")
async def predict_batch(payload: BatchPredictRequest):
    return build_batch_response(payload)


@app.get("/api/metadata")
async def metadata():
    """Endpoint auxiliar para páginas consumirem metadados sem duplicar números."""
    return _metadata_context()
