from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import Optional, List
from src.data_extraction import collect_RAVDESS
from src.data_analysis import analyze_RAVDESS
from src.data_preprocessing import preprocess_dataset, RAVDESS_metadata_generator
import os
import io
import soundfile as sf

app = FastAPI()
RAW_DATA_PATH = "/app/data/raw"
INTERIM_STORAGE_PATH = "/app/data/interim"
PREPROCESEED_DATA_PATH = "/app/data/preprocessed"

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/collect_data")
async def collect_data():

    response = dict()

    RAVDESS_dir = os.path.join(RAW_DATA_PATH, "RAVDESS")
    if os.path.exists(RAVDESS_dir):
        response["RAVDESS"] = "Data already exists."
    else:
        RAVDESS_status = collect_RAVDESS(RAVDESS_dir)
        if RAVDESS_status == 0:
            response["RAVDESS"] = "Data collection successful."
        else:
            response["RAVDESS"] = "Data collection failed."

    ... # Other data collection functions

    return response
    

@app.get("/analyze_data")
async def analyze_data():
    
    response = dict()

    RAVDESS_dir = os.path.join(RAW_DATA_PATH, "RAVDESS")
    RAVDESS_stats_dir = os.path.join(INTERIM_STORAGE_PATH, "RAVDESS")
    if not os.path.exists(RAVDESS_dir):
        response["RAVDESS"] = "Data does not exist. Please collect data first."
    else:
        RAVDESS_features, _ = analyze_RAVDESS(RAVDESS_dir, RAVDESS_stats_dir)
        response["RAVDESS"] = f"Data analysis successful. {len(RAVDESS_features)} files processed."

    ... # Other data analysis functions

    return response


@app.get("/prepare_dataset")
async def prepare_dataset():
    
    response = dict()

    RAVDESS_dir = os.path.join(RAW_DATA_PATH, "RAVDESS")
    try:
        preprocess_dataset(RAVDESS_dir, PREPROCESEED_DATA_PATH, RAVDESS_metadata_generator)
        response["RAVDESS"] = "Data preprocessing successful."
    except Exception as e:
        response["RAVDESS"] = f"Data preprocessing failed. Error: {e}"

    ... # Other data preprocessing functions

    return response

@app.post("/preprocess_audio")
async def preprocess_audio(audio_input: bytes = Body(...)):
    if not audio_input.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    try:
        file_bytes = await audio_input.read()
        
        processed_audio, sr = preprocess_audio(file_bytes, input_type="bytes")
        
        buffer = io.BytesIO()
        sf.write(buffer, processed_audio, sr, format="WAV")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=preprocessed_{audio_input.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")