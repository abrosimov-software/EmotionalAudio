from pydub import AudioSegment
from pydub.effects import normalize
import os
from microservice.src.feature_extraction import extract_labels, extract_audio_features
import numpy as np
import torch

def prepare_data(data: dict, user_data: bool):
    normalized_data = [] 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, 'data/preprocessed/audiofiles')

    for folder in data.keys:
        for filename in folder.keys:
            audio = data[folder][filename]
            tensor = torch.tensor(np.array(normalize(audio).get_array_of_samples()))

            normalized_data.append(tensor)
            if not user_data:
                output_file_path = os.path.join(audio_dir, f"{filename}.pt")
                torch.save(tensor, output_file_path)
    
    extract_labels(audio_dir)
    extract_audio_features(audio_dir)
    return normalized_data