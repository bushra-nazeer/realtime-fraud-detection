from fastapi.testclient import TestClient

from fraud.api.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_rejects_missing_required_field():
    # customer_id is required.
    response = client.post("/score", json={"amount": 50.0})
    assert response.status_code == 422


def test_score_returns_result_or_503():
    payload = {
        "customer_id": 7, "amount": 9000.0, "hour": 3, "merchant_category": "online",
        "is_foreign": 1, "distance_from_home_km": 800.0,
    }
    response = client.post("/score", json=payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_band"] in {"Low", "Medium", "High"}
