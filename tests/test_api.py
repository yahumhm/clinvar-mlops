from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "ClinVar MLOps" in response.text

def test_predict_endpoint():
    payload = {
        "CHROM": "17",
        "AF_ESP": 0.0045,
        "AF_EXAC": 0.0038,
        "AF_TGP": 0.0051,
        "CADD_PHRED": 25.4,
        "CADD_RAW": 3.65,
        "LoFtool": 0.042,
        "IMPACT": "MODERATE"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability_conflict" in data