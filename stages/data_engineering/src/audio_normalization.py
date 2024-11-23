import numpy as np
import os
from pydub import AudioSegment

def calculate_neutral_audio_stats(data):
    neutral_audio_samples = []
    for filename in data.keys:
        emotion_id = filename.split('-')[2]
        if emotion_id == '01':
            audio = data[filename]
            samples = np.array(audio.get_array_of_samples())
            neutral_audio_samples.extend(samples)

    neutral_mean = np.mean(neutral_audio_samples)
    neutral_var = np.var(neutral_audio_samples)
    return neutral_mean, neutral_var

def normalize_audio(audio, neutral_mean, neutral_var):
    samples = np.array(audio.get_array_of_samples())
    normalized_samples = (samples - neutral_mean) / np.sqrt(neutral_var)
    return normalized_samples
