"""Carregamento único do artefato e metadados do modelo."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "no_show_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


class ModelLoadError(RuntimeError):
    """Erro ao carregar artefato ou metadados do modelo."""


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    """Carrega metadados do modelo uma única vez."""
    if not METADATA_PATH.exists():
        raise ModelLoadError(f"Arquivo de metadados não encontrado: {METADATA_PATH}")
    with METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_model():
    """Carrega o modelo treinado uma única vez."""
    if not MODEL_PATH.exists():
        raise ModelLoadError(f"Artefato do modelo não encontrado: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_feature_order() -> list[str]:
    """Retorna ordem das features a partir dos metadados."""
    metadata = load_metadata()
    features = metadata.get("features")
    if not features:
        raise ModelLoadError("Metadados não contêm a lista de features.")
    return list(features)


def model_status() -> dict[str, Any]:
    """Retorna status de carregamento para o endpoint de saúde."""
    try:
        metadata = load_metadata()
        load_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": metadata.get("model_name", "desconhecido"),
            "model_version": metadata.get("model_version", "desconhecido"),
        }
    except ModelLoadError as exc:
        return {
            "status": "error",
            "model_loaded": False,
            "model_name": None,
            "model_version": None,
            "message": str(exc),
        }
