from fastapi.testclient import TestClient
from mlopsproject.api import app
import pytest

client = TestClient(app)

@pytest.mark.skip()
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Face Emotions prediction model inference API!"}

# add another test
@pytest.mark.skip()
def test_invalid_endpoint():
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

@pytest.mark.skip()
def test_post_predict_no_file():
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity due to missing file