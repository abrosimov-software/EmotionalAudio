import os
import torch
import librosa
import numpy as np

def extract_labels(audiofiles_dir):
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

    for filename in os.listdir(audiofiles_dir):
        parts = filename[:-4].split('-')
        emotion = emotion_map.get(parts[2], 'unknown')

        tensor = torch.tensor(emotion)
        torch.save(tensor, os.path.join(audiofiles_dir, f"../labels/{filename}.pt"))
    
    return 0

def extract_audio_features(audiofiles_dir):
    for filename in os.listdir(audiofiles_dir):
        y, sr = librosa.load(filename, sr=None)
        
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

        tensor = torch.tensor(features)
        torch.save(tensor, os.path.join(audiofiles_dir, f"../features/{filename}.pt"))
    return 0