"""Schemas Pydantic para validação de entrada da API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(..., ge=0, le=120, description="Idade do paciente em anos")
    waiting_days: int = Field(..., ge=0, le=200, description="Dias entre agendamento e consulta")
    gender: Literal["F", "M"] = Field(..., description="Gênero: F ou M")
    scholarship: Literal[0, 1] = Field(..., description="Bolsa Família: 0=Não, 1=Sim")
    hypertension: Literal[0, 1] = Field(..., description="Hipertensão: 0=Não, 1=Sim")
    diabetes: Literal[0, 1] = Field(..., description="Diabetes: 0=Não, 1=Sim")
    alcoholism: Literal[0, 1] = Field(..., description="Alcoolismo: 0=Não, 1=Sim")
    handicap: int = Field(..., ge=0, le=4, description="Quantidade de deficiências (0 a 4)")
    sms_received: Literal[0, 1] = Field(..., description="SMS recebido: 0=Não, 1=Sim")

    @field_validator(
        "age",
        "waiting_days",
        "handicap",
        mode="before",
    )
    @classmethod
    def reject_invalid_numbers(cls, value):
        if isinstance(value, bool):
            raise ValueError("Valor booleano não é permitido para este campo.")
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Valor numérico inválido.")
            if not value.is_integer():
                raise ValueError("O valor deve ser um número inteiro.")
            value = int(value)
        return value


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: list[PatientInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de pacientes da agenda (máximo 100)",
    )
    safety_factor: float = Field(0.85, ge=0.0, le=1.0, description="Fator de segurança da política")
    max_extra_percentage: float = Field(
        0.15,
        ge=0.0,
        le=0.30,
        description="Teto percentual de encaixes extras",
    )

    @field_validator("safety_factor", "max_extra_percentage", mode="before")
    @classmethod
    def reject_invalid_floats(cls, value):
        if isinstance(value, bool):
            raise ValueError("Valor booleano não é permitido.")
        if isinstance(value, (int, float)):
            fv = float(value)
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError("Valor numérico inválido.")
            return fv
        raise ValueError("Valor numérico inválido.")
