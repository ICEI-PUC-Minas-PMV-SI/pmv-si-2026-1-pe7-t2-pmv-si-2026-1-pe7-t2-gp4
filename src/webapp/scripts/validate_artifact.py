#!/usr/bin/env python3
"""Valida compatibilidade entre artefato, metadados e pipeline de inferência."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

WEBAPP_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WEBAPP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "no_show_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


def main() -> int:
  errors: list[str] = []

  if not MODEL_PATH.exists():
    errors.append(f"Modelo ausente: {MODEL_PATH}")
  if not METADATA_PATH.exists():
    errors.append(f"Metadados ausentes: {METADATA_PATH}")
  if errors:
    for error in errors:
      print(f"ERRO: {error}")
    return 1

  with METADATA_PATH.open(encoding="utf-8") as handle:
    metadata = json.load(handle)

  model = joblib.load(MODEL_PATH)
  features = metadata.get("features", [])
  if not features:
    errors.append("Lista de features vazia nos metadados.")

  sample = pd.DataFrame([[35, 20, 0, 0, 0, 0, 0, 0, 1]], columns=features)

  try:
    probabilities = model.predict_proba(sample)
    probability = float(probabilities[0][1])
    if not 0.0 <= probability <= 1.0:
      errors.append(f"Probabilidade fora do intervalo [0, 1]: {probability}")
  except Exception as exc:  # pragma: no cover - validação explícita
    errors.append(f"Falha na inferência de teste: {exc}")

  required_metric_keys = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]
  metrics = metadata.get("metrics", {})
  for key in required_metric_keys:
    if key not in metrics:
      errors.append(f"Métrica ausente nos metadados: {key}")

  if errors:
    for error in errors:
      print(f"ERRO: {error}")
    return 1

  print("Validação do artefato concluída com sucesso.")
  print(f"Modelo: {metadata.get('model_name')} v{metadata.get('model_version')}")
  print(f"Features: {', '.join(features)}")
  print(f"PR-AUC: {metrics['pr_auc']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
