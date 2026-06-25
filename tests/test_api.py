import io
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="module")
def client():
    with patch("api.load_model", return_value=MagicMock()):
        from fastapi.testclient import TestClient
        import api
        with TestClient(api.app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_returns_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]


def test_predict_missing_file_returns_422(client):
    r = client.post("/predict")
    assert r.status_code == 422


def test_predict_success(client):
    with patch("api.run_inference", return_value=("Violence", [0.1, 0.9])):
        r = client.post(
            "/predict",
            files={"file": ("clip.mp4", io.BytesIO(b"fake-video"), "video/mp4")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "Violence"
    assert body["confidence"] == 0.9
    assert set(body["probabilities"].keys()) == {"NonViolence", "Violence"}
    assert "latency_seconds" in body


def test_predict_nonviolence(client):
    with patch("api.run_inference", return_value=("NonViolence", [0.95, 0.05])):
        r = client.post(
            "/predict",
            files={"file": ("clip.mp4", io.BytesIO(b"fake-video"), "video/mp4")},
        )
    assert r.status_code == 200
    assert r.json()["prediction"] == "NonViolence"
    assert r.json()["confidence"] == 0.95


def test_predict_short_video_returns_422(client):
    with patch("api.run_inference", return_value=(None, None)):
        r = client.post(
            "/predict",
            files={"file": ("short.mp4", io.BytesIO(b"fake-video"), "video/mp4")},
        )
    assert r.status_code == 422
    assert "too short" in r.json()["detail"].lower()
