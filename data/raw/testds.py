import os
import torch
import numpy as np
from typing import Optional 
from pydub import AudioSegment
import librosa


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

def extract_data(source: Optional[str] = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if source is None:
        audiofiles_dir = os.path.join(current_dir, './test/audiofiles')
    else:
        audiofiles_dir = os.path.join(current_dir, f'data/{source}/audiofiles')

    dataset = {}
    for filename in os.listdir(audiofiles_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(audiofiles_dir, filename)
            dataset[filename] = AudioSegment.from_file(filepath)

    print(f"Extracted {len(dataset)} audio files from {audiofiles_dir}")
    return dataset


def calculate_neutral_audio_stats(data):
    neutral_audio_samples = []
    for filename in data.keys():
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


def extract_labels(audiofiles_dict, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)
    for filename in audiofiles_dict.keys():
        emotion_id = filename.split('-')[2]
        emotion = emotion_map.get(emotion_id, 'unknown')
        label_tensor = torch.tensor([list(emotion_map.values()).index(emotion)])
        torch.save(label_tensor, os.path.join(labels_dir, f"{filename}.pt"))

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

def extract_audio_features(audiofiles_dict, features_dir):
    os.makedirs(features_dir, exist_ok=True)
    for filename, tensor in audiofiles_dict.items():
        features = extract_features_from_file(tensor)
        torch.save(features, os.path.join(features_dir, f"{filename}.pt"))


def prepare_data(data: dict, user_data: bool):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_audio_dir = os.path.join(current_dir, 'data/preprocessed/audiofiles')
    labels_dir = os.path.join(current_dir, 'data/preprocessed/labels')
    features_dir = os.path.join(current_dir, 'data/preprocessed/features')

    os.makedirs(preprocessed_audio_dir, exist_ok=True)

    normalized_data = {}
    neutral_mean, neutral_var = calculate_neutral_audio_stats(data)

    for filename, audio in data.items():
        normalized_audio = normalize_audio(audio, neutral_mean, neutral_var)
        audio_tensor = torch.tensor(normalized_audio)
        torch.save(audio_tensor, os.path.join(preprocessed_audio_dir, f"{filename}.pt"))
        normalized_data[filename] = audio_tensor

    extract_labels(normalized_data, labels_dir)
    extract_audio_features(normalized_data, features_dir)

    print("Data preparation complete.")
    return normalized_data


if __name__ == "__main__":
    dataset = extract_data()
    prepare_data(dataset, user_data=False)
