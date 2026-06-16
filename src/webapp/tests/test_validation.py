from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_negative_age_rejected():
    response = client.post(
        "/api/predict",
        json={
            "age": -1,
            "waiting_days": 10,
            "gender": "F",
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "sms_received": 0,
        },
    )
    assert response.status_code == 422


def test_age_above_120_rejected():
    response = client.post(
        "/api/predict",
        json={
            "age": 121,
            "waiting_days": 10,
            "gender": "M",
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "sms_received": 0,
        },
    )
    assert response.status_code == 422


def test_negative_waiting_days_rejected():
    response = client.post(
        "/api/predict",
        json={
            "age": 30,
            "waiting_days": -1,
            "gender": "F",
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "sms_received": 0,
        },
    )
    assert response.status_code == 422


def test_invalid_binary_rejected():
    response = client.post(
        "/api/predict",
        json={
            "age": 30,
            "waiting_days": 10,
            "gender": "F",
            "scholarship": 2,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "sms_received": 0,
        },
    )
    assert response.status_code == 422


def test_handicap_above_4_rejected():
    response = client.post(
        "/api/predict",
        json={
            "age": 30,
            "waiting_days": 10,
            "gender": "F",
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 5,
            "sms_received": 0,
        },
    )
    assert response.status_code == 422


def test_extra_fields_rejected():
    response = client.post(
        "/api/predict",
        json={
            "age": 30,
            "waiting_days": 10,
            "gender": "F",
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "sms_received": 0,
            "cpf": "123",
        },
    )
    assert response.status_code == 422
