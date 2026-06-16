from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"]
    assert data["model_version"] == "1.0.0"
