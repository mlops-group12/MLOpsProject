from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import torch
from mlopsproject.model import CNN
import sys
import numpy as np
from google.cloud import storage

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Emotions prediction model inference API!"}

@app.post("/predict/")
async def predict(data: UploadFile = File(...)):
    """Predict emotions for an image."""
    i_image = Image.open(data.file)
    if i_image.mode != "L": # RGB to greyscale conversion
        i_image = i_image.convert(mode="L")

    res = i_image.resize((64, 64))

    model = CNN()

    # setup WandB logger only if enabled
    gcs_model_path = None
    if cfg.gcs.bucket and cfg.gcs.model_folder:
        client = storage.Client(project="active-premise-484209-h0")
        bucket = client.bucket(cfg.gcs.bucket)
        blobs = list(bucket.list_blobs(prefix=cfg.gcs.model_folder))
        model_blobs = [b for b in blobs if b.name.endswith(".pt")]
        if model_blobs:
            latest_blob = max(model_blobs, key=lambda b: b.name)
            tmp_dir = tempfile.mkdtemp()
            gcs_model_path = os.path.join(tmp_dir, os.path.basename(latest_blob.name))
            print(f"Downloading latest model from GCS: {latest_blob.name} -> {gcs_model_path}")
            latest_blob.download_to_filename(gcs_model_path)
    
    if local_model_path and os.path.exists(local_model_path):
        print(f"Loading latest local model: {local_model_path}")
        model.load_state_dict(torch.load(local_model_path))
        model_version = os.path.basename(local_model_path).split("_")[-1].replace(".pt", "")
    elif gcs_model_path:
        print(f"Loading latest GCS model: {gcs_model_path}")
        model.load_state_dict(torch.load(gcs_model_path))
        model_version = os.path.basename(gcs_model_path).split("_")[-1].replace(".pt", "")
    else:
        print("No model found locally or in GCS!")
        sys.exit(1)

    """
    try:
        model.load_state_dict(torch.load("models/model_weights_latest.pt"))
    except FileNotFoundError:
        print("Model weights not found!")
        sys.exit(0)
    """
    
    model.eval()
    with torch.no_grad():
        img_np = np.array(res, dtype=np.float32)
        tensor_image = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        output = model(tensor_image)
        predicted = torch.argmax(output.squeeze())

    class_names = ["angry", "fear", "happy", "sad", "surprise"]

    return {"predicted_emotion": class_names[predicted.item()]}

