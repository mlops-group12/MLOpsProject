from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image
import torch
from mlopsproject.model import CNN
import numpy as np
from google.cloud import storage
from datetime import datetime, UTC
import os
import os
import tempfile
import hydra
from omegaconf import DictConfig

# -------------------------
# Initialize Hydra config and help functons
# -------------------------
cfg = hydra.compose(config_name="base_config", config_path="../../configs")

app = FastAPI()
DB_FILE = "prediction_database.csv"

def extract_image_features(image: Image.Image):
    img = np.array(image, dtype=np.float32)

    height, width = img.shape
    brightness = float(np.mean(img))
    contrast = float(np.std(img))

    # Sharpness via gradient magnitude
    gx, gy = np.gradient(img)
    sharpness = float(np.mean(np.sqrt(gx**2 + gy**2)))

    return width, height, brightness, contrast, sharpness

def add_to_database(
    now: str,
    image_name: str,
    width: int,
    height: int,
    brightness: float,
    contrast: float,
    sharpness: float,
    predicted_emotion: str,
):
    with open(DB_FILE, "a") as f:
        f.write(
            f"{now},{image_name},{width},{height},"
            f"{brightness:.4f},{contrast:.4f},{sharpness:.4f},"
            f"{predicted_emotion}\n"
        )


# -------------------------
# Load model from GCS at startup
# -------------------------
model = CNN()
model_version = None

if cfg.gcs.bucket and cfg.gcs.model_folder:
    client = storage.Client(project="active-premise-484209-h0")
    bucket = client.bucket(cfg.gcs.bucket)
    blobs = list(bucket.list_blobs(prefix=cfg.gcs.model_folder))
    model_blobs = [b for b in blobs if b.name.endswith(".pt")]

    if model_blobs:
        # Pick latest model by timestamp in filename
        latest_blob = max(model_blobs, key=lambda b: b.name)
        tmp_dir = tempfile.mkdtemp()
        gcs_model_path = os.path.join(tmp_dir, os.path.basename(latest_blob.name))
        print(f"Downloading latest model from GCS: {latest_blob.name} -> {gcs_model_path}")
        latest_blob.download_to_filename(gcs_model_path)
        model.load_state_dict(torch.load(gcs_model_path))
        model_version = os.path.basename(gcs_model_path).split("_")[-1].replace(".pt", "")
    else:
        raise RuntimeError("No models found in GCS bucket.")
else:
    raise RuntimeError("GCS bucket or model folder not configured.")

model.eval()

@app.on_event("startup")
def startup_event():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            f.write(
                "time,image_name,width,height,brightness,contrast,sharpness,"
                "predicted_emotion\n"
            )

# -------------------------
# API endpoints
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Emotions prediction model inference API!"}

@app.post("/predict/")
async def predict(data: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Predict emotions for an uploaded image."""
    i_image = Image.open(data.file)
    if i_image.mode != "L":
        i_image = i_image.convert(mode="L")
    res = i_image.resize((64, 64))

    with torch.no_grad():
        img_np = np.array(res, dtype=np.float32)
        tensor_image = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        output = model(tensor_image)
        predicted = torch.argmax(output.squeeze())
        class_names = ["angry", "fear", "happy", "sad", "surprise"]
        predicted_emotion = class_names[predicted.item()]

        # extract features for logging
        width, height, brightness, contrast, sharpness = extract_image_features(res)
        now = str(datetime.now(tz=UTC))
        background_tasks.add_task(
            add_to_database,
            now,
            data.filename,
            width,
            height,
            brightness,
            contrast,
            sharpness,
            predicted_emotion,
        )

    return {"predicted_emotion": predicted_emotion, "model_version": model_version}

