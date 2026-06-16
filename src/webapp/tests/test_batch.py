from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PATIENT = {
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


def test_empty_batch_rejected():
    response = client.post(
        "/api/predict-batch",
        json={"patients": [], "safety_factor": 0.85, "max_extra_percentage": 0.15},
    )
    assert response.status_code == 422


def test_batch_over_100_rejected():
    patients = [PATIENT.copy() for _ in range(101)]
    response = client.post(
        "/api/predict-batch",
        json={"patients": patients, "safety_factor": 0.85, "max_extra_percentage": 0.15},
    )
    assert response.status_code == 422


def test_batch_expected_absences_and_policy():
    patients = [PATIENT, {**PATIENT, "age": 50, "waiting_days": 5, "gender": "M"}]
    response = client.post(
        "/api/predict-batch",
        json={"patients": patients, "safety_factor": 0.85, "max_extra_percentage": 0.15},
    )
    assert response.status_code == 200
    data = response.json()
    probabilities = [item["probability"] for item in data["patients"]]
    expected = round(sum(probabilities), 6)
    assert data["policy"]["expected_absences"] == expected
    assert data["policy"]["safety_factor"] == 0.85
    assert data["policy"]["hard_cap"] == 0
    assert data["policy"]["recommended_extra_slots"] <= data["policy"]["hard_cap"] or data["policy"]["hard_cap"] == 0


def test_batch_recommendation_respects_cap():
    patients = [PATIENT.copy() for _ in range(20)]
    response = client.post(
        "/api/predict-batch",
        json={"patients": patients, "safety_factor": 1.0, "max_extra_percentage": 0.15},
    )
    assert response.status_code == 200
    policy = response.json()["policy"]
    assert policy["recommended_extra_slots"] <= policy["hard_cap"]
    assert policy["recommended_extra_slots"] >= 0
