import os
from typing import Optional, List
from pydub import AudioSegment
import requests
import zipfile
import io

def collect_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audiofiles_dir = os.path.join(current_dir, 'data/raw/audiofiles')

    archive1 = requests.get("https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1").content
    archive2 = requests.get("https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1").content
    try:
        for archive in [archive1, archive2]:
            with zipfile.ZipFile(io.BytesIO(archive)) as zip_ref:
                for folder in zip_ref.infolist():
                    zip_ref.extract(folder, audiofiles_dir)
        return 0
    except:
        return 1


def extract_data(source: Optional[str] = None):
    if source == None:
        audiofiles_dir = os.path.join('data/raw/audiofiles')
    else:
        audiofiles_dir = os.path.join(f'data/{source}/audiofiles')
    
    dataset = {}
    for folder in os.listdir(audiofiles_dir):
        dataframe = {}
        for filename in os.listdir(folder):
            dataframe[filename] = AudioSegment.from_file(filename)
        dataset[folder] = dataframe
    
    return dataset