import os
import torch
import librosa
import numpy as np
from microservice.src.audio_normalization import calculate_neutral_stats, normalize_features

def extract_labels(audiofiles_dict, audiofiles_dir):
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    for filename in audiofiles_dict.keys:
        parts = filename[:-4].split('-')
        emotion = emotion_map.get(parts[2], 'unknown')

        tensor = torch.tensor(emotion)
        torch.save(tensor, os.path.join(audiofiles_dir, f"../labels/{filename}.pt"))
    
    return 0

def extract_features_from_file(audio_tensor):
    y = audio_tensor.numpy()
    sr = 16000

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

def extract_audio_features(audiofiles_dict, audiofiles_dir):
    for filename in audiofiles_dict.keys:
        tensor = audiofiles_dict[filename]
        features = extract_features_from_file(tensor)
        output_tensor = torch.tensor(features)

        output_file_path = os.path.join(audiofiles_dir, f"../features/{filename}.pt")
        torch.save(output_tensor, output_file_path)
    return 0