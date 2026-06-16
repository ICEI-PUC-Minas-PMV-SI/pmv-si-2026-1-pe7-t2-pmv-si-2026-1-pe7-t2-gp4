"""Política conservadora de overbooking baseada em probabilidades."""

from __future__ import annotations

import math
from typing import Sequence


def risk_band(probability: float) -> str:
    """Classifica faixa de risco apenas para comunicação visual."""
    if probability < 0.40:
        return "baixo"
    if probability < 0.70:
        return "moderado"
    return "alto"


def calculate_overbooking(
    probabilities: Sequence[float],
    safety_factor: float = 0.85,
    max_extra_percentage: float = 0.15,
) -> dict[str, float | int]:
    """
    Calcula recomendação de encaixes extras.

    expected_absences = soma das probabilidades
    preliminary_slots = floor(expected_absences × safety_factor)
    hard_cap = floor(quantidade × max_extra_percentage)
    recommended = min(preliminary_slots, hard_cap)
    """
    patient_count = len(probabilities)
    expected_absences = float(sum(probabilities))
    preliminary_slots = int(math.floor(expected_absences * safety_factor))
    hard_cap = int(math.floor(patient_count * max_extra_percentage))
    recommended_extra_slots = max(0, min(preliminary_slots, hard_cap))

    return {
        "patient_count": patient_count,
        "expected_absences": round(expected_absences, 6),
        "preliminary_slots": preliminary_slots,
        "hard_cap": hard_cap,
        "recommended_extra_slots": recommended_extra_slots,
        "safety_factor": safety_factor,
        "max_extra_percentage": max_extra_percentage,
    }
