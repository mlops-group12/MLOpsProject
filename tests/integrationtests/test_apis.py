from fastapi.testclient import TestClient
from mlopsproject.api import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Face Emotions prediction model inference API!"}