from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = FastAPI()

# Load the pretrained model and feature extractor
model = AutoModelForAudioClassification.from_pretrained("/app/models/distilhubert")
model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained("ntu-spml/distilhubert")

# Access the label mappings from the model's config
id2label = model.config.id2label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate the uploaded file
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV and MP3 are accepted.")

    try:
        # Read the uploaded file into memory
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Load and preprocess the audio file using the feature extractor
        inputs = feature_extractor(audio_stream, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            # Make prediction
            logits = model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            predicted_label = id2label[predicted_id]

        # Return the prediction
        return JSONResponse(content={"prediction": predicted_label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))