from pydub import AudioSegment
from pydub.effects import normalize
import os

def process_data(data: dict):
    normalized_data = [] 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    audiofiles_dir = os.path.join(current_dir, '../../../data/preprocessed/audiofiles')

    for filename in data.keys:
        audio = data[filename]
        normalized_data.append(normalize(audio))

        output_file_path = os.path.join(audiofiles_dir, f"{filename}.wav")
        audio.export(output_file_path, format='wav')

    return normalized_data