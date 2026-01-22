from fastapi.testclient import TestClient
import pytest
from mlopsproject.api import app
from io import BytesIO
from PIL import Image

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Face Emotions prediction model inference API!"}

# add another test
def test_invalid_endpoint():
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

def test_post_predict_no_file():
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity due to missing file

@pytest.mark.skip(reason="Requires valid model and GCS setup")
def test_post_predict_with_file():
    # create a dummy image file for testing

    img = Image.new('L', (64, 64), color = 'white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    files = {'data': ('test.png', img_byte_arr, 'image/png')}
    response = client.post("/predict/", files=files)
    
    assert response.status_code == 200
    
    json_response = response.json()

    assert "predicted_emotion" in json_response
    assert "model_version" in json_response
    assert "probs" in json_response