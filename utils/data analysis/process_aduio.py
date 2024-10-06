import os
import pandas as pd
import librosa
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
audiofiles_dir = os.path.join(current_dir, '../../data/raw/audiofiles')

df = pd.read_csv(os.path.join(current_dir, '../../data/raw/audiofile_dataset.csv'))

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    
    features = {
        'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
        'mfcc_var': np.var(mfccs, axis=1).tolist(),
        'chroma_mean': np.mean(chroma, axis=1).tolist(),
        'chroma_var': np.var(chroma, axis=1).tolist(),
        'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist(),
        'zcr_mean': np.mean(zcr).item(),
        'rms_mean': np.mean(rms).item(),
        'tempo': tempo,
        'duration': duration
    }
    
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
        audio_features = extract_audio_features(file_path)
        flat_features = flatten_features(audio_features)
        audio_features_list.append(flat_features)
    else:
        print(f"File {file_path} not found!")

audio_features_df = pd.DataFrame(audio_features_list)

df_with_merged_features = pd.concat([df, audio_features_df], axis=1)

df_with_merged_features.to_csv(os.path.join(current_dir, '../../data/raw/audiofile_dataset_with_features.csv'), index=False)