from pydub import AudioSegment
from pydub.effects import normalize
import os

def prepare_data(data: dict, user_data: bool):
    normalized_data = [] 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    audiofiles_dir = os.path.join(current_dir, 'data/preprocessed/audiofiles')

    for folder in data.keys:
        for filename in folder.keys:
            audio = data[folder][filename]
            normalized_data.append(normalize(audio))
            if not user_data:
                output_file_path = os.path.join(audiofiles_dir, f"{folder/filename}.wav")
                audio.export(output_file_path, format='wav')

    return normalized_data