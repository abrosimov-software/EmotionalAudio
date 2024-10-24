import os
from typing import Optional, List
from pydub import AudioSegment

def extract_data(source: Optional[str] = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if source == None:
        audiofiles_dir = os.path.join(current_dir, '../../../data/raw/audiofiles')
    else:
        audiofiles_dir = os.path.join(current_dir, f'../../../data/{source}/audiofiles')
    
    dataset = {}
    for filename in os.listdir(audiofiles_dir):
        dataset[filename] = AudioSegment.from_file(filename)
    
    return dataset