import os
import pandas as pd
import pywt
import numpy as np
import librosa

audiofiles_dir = 'audiofiles'

df = pd.read_csv('audiofile_dataset.csv')

def extract_wavelet_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    coeffs = pywt.wavedec(y, wavelet='haar', level=5)
    features = {}

    for i, coeff in enumerate(coeffs):
        features[f'wavelet_coeff_{i}_mean'] = np.mean(coeff)
        features[f'wavelet_coeff_{i}_var'] = np.var(coeff)
    
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    
    return features

def flatten_features(features):
    flat_features = {}
    for key, value in features.items():
        if isinstance(value, list):
            for i, v in enumerate(value):
                flat_features[f'{key}_{i}'] = v
        else:
            flat_features[key] = value
    return flat_features

audio_features_list = []

for filename in df['id']:
    file_path = os.path.join(audiofiles_dir, filename)
    if os.path.exists(file_path):
        audio_features = extract_wavelet_features(file_path)
        flat_features = flatten_features(audio_features)
        audio_features_list.append(flat_features)
    else:
        print(f"File {file_path} not found!")

audio_features_df = pd.DataFrame(audio_features_list)

df_with_merged_features = pd.concat([df, audio_features_df], axis=1)

df_with_merged_features.to_csv('audiofile_dataset_with_wavelet_features.csv', index=False)
