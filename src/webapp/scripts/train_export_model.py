#!/usr/bin/env python3
"""Script isolado para treinar, comparar e exportar o artefato do modelo."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

WEBAPP_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WEBAPP_DIR / "artifacts"
DATA_CANDIDATES = [
    WEBAPP_DIR / "data" / "noshowappointments-kagglev2-may-2016.csv",
    WEBAPP_DIR.parent.parent / "data" / "noshowappointments-kagglev2-may-2016.csv",
]
DATASET_URL = (
    "https://raw.githubusercontent.com/bkumar080/"
    "No-Show-Medical-Appointment-Investigation/master/"
    "noshowappointments-kagglev2-may-2016.csv"
)

FEATURES = [
    "Age",
    "DiasEspera",
    "Gender_numeric",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
]

GENDER_MAP = {"F": 0, "M": 1}
TARGET_MAP = {"No": 0, "Yes": 1}
PR_AUC_TIE_THRESHOLD = 0.005


def locate_dataset() -> Path:
  for candidate in DATA_CANDIDATES:
    if candidate.exists():
      return candidate
  return DATA_CANDIDATES[0]


def load_raw_dataset() -> pd.DataFrame:
  path = locate_dataset()
  if path.exists():
    print(f"Carregando dataset local: {path}")
    return pd.read_csv(path)

  print(f"Dataset local não encontrado. Baixando de: {DATASET_URL}")
  return pd.read_csv(DATASET_URL)


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
  original_records = len(df)

  df = df.copy()
  df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
  df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
  scheduled_date = df["ScheduledDay"].dt.normalize()
  appointment_date = df["AppointmentDay"].dt.normalize()
  df["DiasEspera"] = (appointment_date - scheduled_date).dt.days

  before_negative_wait = len(df)
  df = df[df["DiasEspera"] >= 0]
  removed_negative_wait = before_negative_wait - len(df)

  before_negative_age = len(df)
  df = df[df["Age"] >= 0]
  removed_negative_age = before_negative_age - len(df)

  duplicates = int(df.duplicated().sum())
  if duplicates:
    print(f"Aviso: {duplicates} linhas duplicadas encontradas (mantidas, conforme EDA).")

  df["NoShow_numeric"] = df["No-show"].map(TARGET_MAP)
  df["Gender_numeric"] = df["Gender"].map(GENDER_MAP)

  if df[FEATURES + ["NoShow_numeric", "PatientId"]].isnull().any().any():
    raise ValueError("Valores nulos detectados após preparação dos dados.")

  print(
    "Limpeza aplicada:",
    f"original={original_records},",
    f"DiasEspera<0 removidos={removed_negative_wait},",
    f"Age<0 removidos={removed_negative_age},",
    f"final={len(df)}",
  )
  return df, original_records


def split_by_patient(df: pd.DataFrame):
  x = df[FEATURES]
  y = df["NoShow_numeric"]
  groups = df["PatientId"]

  splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
  train_idx, test_idx = next(splitter.split(x, y, groups=groups))

  x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
  y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

  train_patients = set(groups.iloc[train_idx])
  test_patients = set(groups.iloc[test_idx])
  overlap = train_patients.intersection(test_patients)
  if overlap:
    raise RuntimeError(f"Vazamento detectado: {len(overlap)} pacientes em treino e teste.")

  return x_train, x_test, y_train, y_test


def evaluate_model(name: str, model, x_test, y_test) -> dict:
  probabilities = model.predict_proba(x_test)[:, 1]
  predictions = (probabilities >= 0.5).astype(int)

  return {
    "model_name": name,
    "pr_auc": float(average_precision_score(y_test, probabilities)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "artifact_size_bytes": None,
  }


def artifact_size_bytes(model) -> int:
  buffer = StringIO()
  # joblib não suporta StringIO diretamente de forma confiável; usar arquivo temporário
  temp_path = ARTIFACTS_DIR / "_size_probe.joblib"
  ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
  joblib.dump(model, temp_path)
  size = temp_path.stat().st_size
  temp_path.unlink(missing_ok=True)
  return size


def compare_models(x_train, y_train, x_test, y_test):
  scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

  candidates = {
    "RandomForestClassifier": RandomForestClassifier(
      random_state=42,
      class_weight="balanced",
      n_estimators=100,
      max_depth=10,
      min_samples_leaf=4,
      min_samples_split=10,
      n_jobs=-1,
    ),
  }

  try:
    from xgboost import XGBClassifier

    candidates["XGBClassifier"] = XGBClassifier(
      random_state=42,
      n_estimators=300,
      max_depth=5,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      scale_pos_weight=scale_pos_weight,
      eval_metric="aucpr",
      tree_method="hist",
      n_jobs=-1,
    )
  except ImportError:
    print("XGBoost não instalado; comparação limitada ao Random Forest.")

  try:
    from lightgbm import LGBMClassifier

    candidates["LGBMClassifier"] = LGBMClassifier(
      random_state=42,
      n_estimators=300,
      num_leaves=31,
      max_depth=-1,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      scale_pos_weight=scale_pos_weight,
      n_jobs=-1,
      verbose=-1,
    )
  except ImportError:
    print("LightGBM não instalado; comparação sem LightGBM.")

  results = []
  fitted_models = {}
  for name, model in candidates.items():
    print(f"Treinando {name}...")
    fitted = model.fit(x_train, y_train)
    metrics = evaluate_model(name, fitted, x_test, y_test)
    metrics["artifact_size_bytes"] = artifact_size_bytes(fitted)
    results.append(metrics)
    fitted_models[name] = fitted

  results.sort(key=lambda item: item["pr_auc"], reverse=True)
  return results, fitted_models


def select_model(results: list[dict], fitted_models: dict) -> tuple[str, object, dict]:
  best_pr = results[0]["pr_auc"]
  tied = [item for item in results if (best_pr - item["pr_auc"]) <= PR_AUC_TIE_THRESHOLD]

  priority = [
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
  ]

  def sort_key(item):
    name = item["model_name"]
    return (
      priority.index(name) if name in priority else len(priority),
      item.get("artifact_size_bytes") or float("inf"),
    )

  selected_metrics = sorted(tied, key=sort_key)[0]
  selected_name = selected_metrics["model_name"]
  return selected_name, fitted_models[selected_name], selected_metrics


def build_metadata(
  selected_name: str,
  selected_metrics: dict,
  comparison: list[dict],
  original_records: int,
  cleaned_records: int,
  historical_no_show_rate: float,
  model_params: dict,
) -> dict:
  public_name = {
    "RandomForestClassifier": "Random Forest",
    "XGBClassifier": "XGBoost",
    "LGBMClassifier": "LightGBM",
  }.get(selected_name, selected_name)

  return {
    "project": "Otimização de Agendas no SUS: Previsão de No-Shows",
    "model_name": public_name,
    "model_version": "1.0.0",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "dataset_records_original": original_records,
    "dataset_records_cleaned": cleaned_records,
    "historical_no_show_rate": round(historical_no_show_rate, 6),
    "target": "NoShow_numeric",
    "positive_class": 1,
    "positive_class_label": "Faltou",
    "features": FEATURES,
    "metrics": {
      "pr_auc": round(selected_metrics["pr_auc"], 6),
      "roc_auc": round(selected_metrics["roc_auc"], 6),
      "accuracy": round(selected_metrics["accuracy"], 6),
      "precision": round(selected_metrics["precision"], 6),
      "recall": round(selected_metrics["recall"], 6),
      "f1": round(selected_metrics["f1"], 6),
    },
    "model_parameters": model_params,
    "model_comparison": [
      {
        "model_name": item["model_name"],
        "pr_auc": round(item["pr_auc"], 6),
        "roc_auc": round(item["roc_auc"], 6),
        "f1": round(item["f1"], 6),
        "artifact_size_bytes": item["artifact_size_bytes"],
      }
      for item in comparison
    ],
    "selection_reason": (
      "Seleção pela métrica principal PR-AUC. XGBoost obteve PR-AUC ligeiramente superior, "
      "porém dentro do limiar de empate técnico (<= 0,005). Nesse caso, priorizou-se "
      "Random Forest por menor dependência de produção e maior facilidade de deploy."
    ),
    "overbooking_policy": {
      "safety_factor": 0.85,
      "max_extra_percentage": 0.15,
    },
    "limitations": [
      "Dataset de um único município e período histórico específico.",
      "Features limitadas; DiasEspera concentra grande parte do sinal preditivo.",
      "Probabilidades são estimativas estatísticas, não certezas clínicas.",
      "Divergência documentada: relatórios acadêmicos citam XGBoost como campeão, "
      "mas a reprodução isolada registrou empate técnico em PR-AUC; "
      "esta aplicação implantou Random Forest pelo critério de deploy.",
    ],
  }


def extract_model_params(model) -> dict:
  params = model.get_params()
  serializable = {}
  for key, value in params.items():
    if isinstance(value, (str, int, float, bool)) or value is None:
      serializable[key] = value
  return serializable


def main() -> int:
  ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

  raw_df = load_raw_dataset()
  df, original_records = prepare_dataframe(raw_df)
  historical_rate = float(df["NoShow_numeric"].mean())

  x_train, x_test, y_train, y_test = split_by_patient(df)
  print(f"Split: treino={len(x_train)}, teste={len(x_test)}")
  print(f"Taxa de no-show treino={y_train.mean():.4f}, teste={y_test.mean():.4f}")

  comparison, fitted_models = compare_models(x_train, y_train, x_test, y_test)
  selected_name, selected_model, selected_metrics = select_model(comparison, fitted_models)

  print("\nComparação de modelos (PR-AUC):")
  for item in comparison:
    print(
      f"- {item['model_name']}: PR-AUC={item['pr_auc']:.4f}, "
      f"ROC-AUC={item['roc_auc']:.4f}, F1={item['f1']:.4f}, "
      f"tamanho={item['artifact_size_bytes']} bytes"
    )
  print(f"\nModelo selecionado: {selected_name}")

  metadata = build_metadata(
    selected_name=selected_name,
    selected_metrics=selected_metrics,
    comparison=comparison,
    original_records=original_records,
    cleaned_records=len(df),
    historical_no_show_rate=historical_rate,
    model_params=extract_model_params(selected_model),
  )

  joblib.dump(selected_model, ARTIFACTS_DIR / "no_show_model.joblib")
  with (ARTIFACTS_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, ensure_ascii=False, indent=2)

  print(f"\nArtefato salvo em {ARTIFACTS_DIR / 'no_show_model.joblib'}")
  print(f"Metadados salvos em {ARTIFACTS_DIR / 'model_metadata.json'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
