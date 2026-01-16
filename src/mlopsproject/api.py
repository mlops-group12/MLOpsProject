from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import torch
from model import CNN
import sys
import numpy as np

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
    try:
        model.load_state_dict(torch.load("../../models/model_weights_latest.pt"))
    except FileNotFoundError:
        print("Model weights not found!")
        sys.exit(0)
    
    model.eval()
    with torch.no_grad():
        img_np = np.array(res, dtype=np.float32)
        tensor_image = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        output = model(tensor_image)
        predicted = torch.argmax(output.squeeze())

    class_names = ["angry", "fear", "happy", "sad", "surprise"]

    return {"predicted_emotion": class_names[predicted.item()]}

