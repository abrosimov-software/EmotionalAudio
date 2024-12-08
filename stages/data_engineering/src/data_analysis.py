import os
import numpy as np
import librosa
import pandas as pd

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    # Aggregate statistics
    feature_stats = {
        "chroma_stft_mean": np.mean(chroma_stft),
        "chroma_stft_std": np.std(chroma_stft),
        "rmse_mean": np.mean(rmse),
        "rmse_std": np.std(rmse),
        "spec_cent_mean": np.mean(spec_cent),
        "spec_cent_std": np.std(spec_cent),
        "spec_bw_mean": np.mean(spec_bw),
        "spec_bw_std": np.std(spec_bw),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_std": np.std(rolloff),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
        **{f"mfcc_{i+1}_mean": np.mean(mfcc[i]) for i in range(mfcc.shape[0])},
        **{f"mfcc_{i+1}_std": np.std(mfcc[i]) for i in range(mfcc.shape[0])},
        **{f"spec_contrast_{i+1}_mean": np.mean(spec_contrast[i]) for i in range(spec_contrast.shape[0])},
        **{f"spec_contrast_{i+1}_std": np.std(spec_contrast[i]) for i in range(spec_contrast.shape[0])},
        **{f"tonnetz_{i+1}_mean": np.mean(tonnetz[i]) for i in range(tonnetz.shape[0])},
        **{f"tonnetz_{i+1}_std": np.std(tonnetz[i]) for i in range(tonnetz.shape[0])},
    }
    
    return feature_stats

def parse_RAVDESS_labels(file_name):
    parts = file_name.split("-")
    labels = {
        "modality": int(parts[0]),
        "vocal_channel": int(parts[1]),
        "emotion": int(parts[2]),
        "intensity": int(parts[3]),
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor": int(parts[6].split(".")[0]),
    }
    return labels

def analyze_RAVDESS(RAVDESS_DIR, INTERIM_DIR):
    features_list = []
    file_names = os.listdir(RAVDESS_DIR)

    for file_name in file_names:
        if file_name.endswith(".wav"):
            audio_path = os.path.join(RAVDESS_DIR, file_name)
            
            features = extract_audio_features(audio_path)
            
            labels = parse_RAVDESS_labels(file_name)
            
            features.update(labels)
            features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    eda_stats = features_df.groupby(list(parse_RAVDESS_labels("1-1-1-1-1-1-1.wav").keys())).agg(["mean", "std"]).reset_index()
    features_df["file_name"] = file_names

    features_path = os.path.join(INTERIM_DIR, "RAVDESS_features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"RAVDESS features saved to {features_path}")

    eda_path = os.path.join(INTERIM_DIR, "RAVDESS_eda_stats.csv")
    eda_stats.to_csv(eda_path, index=False)
    print(f"EDA statistics saved to {eda_path}")
    
    return features_df, eda_stats