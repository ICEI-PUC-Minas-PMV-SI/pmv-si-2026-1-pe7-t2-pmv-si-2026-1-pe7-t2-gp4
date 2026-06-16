"""Transformação de dados compatível com o pipeline de treinamento."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.schemas import PatientInput

GENDER_MAP = {"F": 0, "M": 1}


def encode_gender(gender: str) -> int:
    """Converte gênero textual para valor numérico."""
    return GENDER_MAP[gender]


def _row_mapping(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "Age": data["age"],
        "DiasEspera": data["waiting_days"],
        "Gender_numeric": encode_gender(data["gender"]),
        "Scholarship": data["scholarship"],
        "Hipertension": data["hypertension"],
        "Diabetes": data["diabetes"],
        "Alcoholism": data["alcoholism"],
        "Handcap": data["handicap"],
        "SMS_received": data["sms_received"],
    }


def patient_to_features(patient: PatientInput | dict[str, Any], feature_order: list[str]) -> pd.DataFrame:
    """Monta DataFrame de uma linha na ordem definida pelos metadados."""
    if isinstance(patient, PatientInput):
        data = patient.model_dump()
    else:
        data = patient
    mapping = _row_mapping(data)
    return pd.DataFrame([[mapping[name] for name in feature_order]], columns=feature_order)


def patients_to_matrix(patients: list[PatientInput], feature_order: list[str]) -> pd.DataFrame:
    """Monta DataFrame de features para predição em lote."""
    rows = [_row_mapping(patient.model_dump()) for patient in patients]
    return pd.DataFrame(rows, columns=feature_order)[feature_order]
