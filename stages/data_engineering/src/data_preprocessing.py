import librosa
import numpy as np
import os 
import pandas as pd
import soundfile as sf
import io

def preprocess_audio(audio_input, target_sr=16000, input_type="file"):
    if input_type == "file":
        y, sr = librosa.load(audio_input, sr=None)
    elif input_type == "bytes":
        with io.BytesIO(audio_input) as byte_stream:
            y, sr = librosa.load(byte_stream, sr=None)
    else:
        raise ValueError("Invalid input_type. Must be 'file' or 'bytes'.")
    
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    y, _ = librosa.effects.trim(y)
    
    y = librosa.util.normalize(y)
    
    return y, sr

def preprocess_dataset(input_dir, output_dir, metadata_generator, target_sr=16000):
    audio_output_dir = os.path.join(output_dir, "audiofiles")
    metadata_output_path = os.path.join(output_dir, "metadata.csv")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    if os.path.exists(metadata_output_path):
        metadata_df = pd.read_csv(metadata_output_path)
    else:
        metadata_df = pd.DataFrame()

    new_metadata = []
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".wav"):
            continue
        
        audio_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(audio_output_dir, file_name)
        
        if os.path.exists(output_path):
            continue
        
        processed_audio, sr = preprocess_audio(audio_path, target_sr=target_sr)
        
        sf.write(output_path, processed_audio, sr)
        
        metadata_entry = metadata_generator(file_name)
        new_metadata.append(metadata_entry)
    
    if new_metadata:
        new_metadata_df = pd.DataFrame(new_metadata)
        if not metadata_df.empty:
            metadata_df = pd.concat([metadata_df, new_metadata_df], ignore_index=True).drop_duplicates(subset=["file_name"])
        else:
            metadata_df = new_metadata_df
        metadata_df.to_csv(metadata_output_path, index=False)

def RAVDESS_metadata_generator(file_name):
    res = {
        "file_name": file_name,
        "target": int(file_name.split("-")[2]) - 1
    }
    return res