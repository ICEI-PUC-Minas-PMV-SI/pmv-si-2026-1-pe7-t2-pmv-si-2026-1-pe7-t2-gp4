from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "waiting_days": 20,
    "gender": "F",
    "scholarship": 0,
    "hypertension": 0,
    "diabetes": 0,
    "alcoholism": 0,
    "handicap": 0,
    "sms_received": 1,
}


def test_predict_returns_probability_range():
    response = client.post("/api/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["probability_percent"] <= 100.0
    assert data["risk_band"] in {"baixo", "moderado", "alto"}
    assert "Probabilidade estimada" in data["message"]


def test_predict_does_not_claim_certainty():
    response = client.post("/api/predict", json=VALID_PAYLOAD)
    data = response.json()
    forbidden = ["vai faltar", "não comparecerá", "deve ser cancelada"]
    combined = (data["message"] + " " + data.get("disclaimer", "")).lower()
    for phrase in forbidden:
        assert phrase not in combined
