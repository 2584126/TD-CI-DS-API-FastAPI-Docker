from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# 1) Test prédiction correcte
def test_predict_success():
    response = client.post("/predict", json={
        "features": [1.0, 2.0, 3.0]
    })

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert response.json() == {"predictions": [2.0, 4.0, 6.0]}


# 2) Test prédiction incorrecte
def test_predict_incorrect():
    response = client.post("/predict", json={
        "features": [1.0, 2.0, 3.0]
    })

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert response.json() != {"predictions": [10.0, 20.0, 30.0]}


# 3) Test JSON incorrect
def test_predict_unprocessable_entity():
    response = client.post("/predict", json={
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 4.9
    })

    assert response.status_code == 422