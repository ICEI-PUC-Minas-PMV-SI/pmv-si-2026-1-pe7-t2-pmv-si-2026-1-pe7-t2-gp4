from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PAGES = ["/", "/simulador", "/agenda", "/metodologia", "/etica", "/sobre"]


def test_home_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "No-Show" in response.text


def test_all_pages_return_200():
    for path in PAGES:
        response = client.get(path)
        assert response.status_code == 200, path


def test_static_css_is_served():
    response = client.get("/public/css/styles.css")
    assert response.status_code == 200
    assert "color-primary" in response.text


def test_static_js_is_served():
    response = client.get("/public/js/common.js")
    assert response.status_code == 200
    assert "SUSApp" in response.text
