from pydub import AudioSegment
from pydub.effects import normalize
import os
from microservice.src.feature_extraction import extract_labels, extract_audio_features
from audio_normalization import calculate_neutral_audio_stats, normalize_audio
import torch

def prepare_data(data: dict, user_data: bool):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, 'data/preprocessed/audiofiles')

    normalized_data = {}

    for folder in data.keys:
        neutral_mean, neutral_var = calculate_neutral_audio_stats(folder)
        for filename in folder.keys:
            audio = data[folder][filename]
            normalized_audio = normalize_audio(audio, neutral_mean, neutral_var)

            tensor = torch.tensor(normalized_audio)
            normalized_data[filename] = tensor
            if not user_data:
                output_file_path = os.path.join(audio_dir, f"{filename}.pt")
                torch.save(tensor, output_file_path)
            
    
    extract_labels(normalized_data, audio_dir)
    extract_audio_features(normalized_data, audio_dir)

    return normalized_data