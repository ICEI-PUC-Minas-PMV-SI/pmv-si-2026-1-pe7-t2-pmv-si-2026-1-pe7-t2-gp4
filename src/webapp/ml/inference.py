"""Inferência individual e em lote."""

from __future__ import annotations

from ml.model_loader import get_feature_order, load_metadata, load_model
from ml.policy import calculate_overbooking, risk_band
from ml.preprocessing import patient_to_features, patients_to_matrix
from ml.schemas import BatchPredictRequest, PatientInput

DISCLAIMER = (
    "Esta é uma estimativa estatística de apoio à decisão. "
    "Não substitui avaliação humana e não deve ser usada para cancelar consultas."
)

PREDICTION_MESSAGE = "Probabilidade estimada de não comparecimento."


def predict_probability(patient: PatientInput) -> float:
    """Retorna probabilidade da classe positiva (falta)."""
    model = load_model()
    feature_order = get_feature_order()
    features = patient_to_features(patient, feature_order)
    probability = float(model.predict_proba(features)[0][1])
    return max(0.0, min(1.0, probability))


def build_prediction_response(patient: PatientInput) -> dict:
    """Monta resposta do endpoint individual."""
    metadata = load_metadata()
    probability = predict_probability(patient)
    probability_percent = round(probability * 100, 2)

    return {
        "probability": round(probability, 6),
        "probability_percent": probability_percent,
        "risk_band": risk_band(probability),
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "message": PREDICTION_MESSAGE,
        "disclaimer": DISCLAIMER,
    }


def build_batch_response(payload: BatchPredictRequest) -> dict:
    """Monta resposta do endpoint de agenda em lote."""
    metadata = load_metadata()
    model = load_model()
    feature_order = get_feature_order()
    matrix = patients_to_matrix(payload.patients, feature_order)
    probabilities = [max(0.0, min(1.0, float(p))) for p in model.predict_proba(matrix)[:, 1]]

    patient_results = []
    for index, (patient, probability) in enumerate(zip(payload.patients, probabilities), start=1):
        probability_percent = round(probability * 100, 2)
        patient_results.append(
            {
                "index": index,
                "input": patient.model_dump(),
                "probability": round(probability, 6),
                "probability_percent": probability_percent,
                "risk_band": risk_band(probability),
            }
        )

    policy = calculate_overbooking(
        probabilities,
        safety_factor=payload.safety_factor,
        max_extra_percentage=payload.max_extra_percentage,
    )

    return {
        "patients": patient_results,
        "policy": policy,
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "message": PREDICTION_MESSAGE,
        "disclaimer": DISCLAIMER,
        "formula_explanation": (
            "Faltas esperadas = soma das probabilidades individuais. "
            "Encaixes preliminares = piso(faltas esperadas × fator de segurança). "
            "Teto = piso(quantidade de pacientes × percentual máximo). "
            "Recomendação = mínimo entre encaixes preliminares e teto."
        ),
    }
