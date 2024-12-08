from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import io
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import os
import librosa  # Add this import

app = FastAPI()

models_dir = "/app/models"
available_models = ["wav2vec2-base", "distilhubert"]

model_dict = {}

# Load models and feature extractors
for model_name in available_models:
    model_path = os.path.join(models_dir, model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    model_dict[model_name] = {
        "feature_extractor": feature_extractor,
        "model": model
    }

def predict(model, feature_extractor, audio_bytes):
    # Load audio data from bytes
    audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    # Prepare inputs
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt")
    # Run prediction
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]
    return predicted_label

@app.get("/")
async def root():
    return {"message": "Welcome to the audio classification API!"}

@app.get("/models")
async def get_models():
    return {"models": available_models}

@app.post("/predict")
async def predict_endpoint(
    model_name: str = Query(...),
    file: UploadFile = File(...)
):
    # Validate model name
    if model_name not in available_models:
        raise HTTPException(status_code=400, detail="Invalid model name.")

    # Validate file type
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV and MP3 are accepted.")

    try:
        # Read the audio bytes
        audio_bytes = await file.read()

        # Get the model and feature extractor
        selected_model = model_dict[model_name]["model"]
        feature_extractor = model_dict[model_name]["feature_extractor"]

        # Make prediction
        predicted_label = predict(selected_model, feature_extractor, audio_bytes)

        return JSONResponse(content={"prediction": predicted_label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))