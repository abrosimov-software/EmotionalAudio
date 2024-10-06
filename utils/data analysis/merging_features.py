import os
import pandas as pd
import librosa
import numpy as np

audio_features_df = pd.read_csv('audiofile_dataset_with_features.csv.csv')

mfcc_means = audio_features_df.filter(like='mfcc_mean').mean(axis=1)
mfcc_vars = audio_features_df.filter(like='mfcc_var').mean(axis=1)
chroma_means = audio_features_df.filter(like='chroma_mean').mean(axis=1)
chroma_vars = audio_features_df.filter(like='chroma_var').mean(axis=1)
spectral_contrast_means = audio_features_df.filter(like='spectral_contrast_mean').mean(axis=1)

merged_features_df = pd.DataFrame({
    'mfcc_mean': mfcc_means,
    'mfcc_var': mfcc_vars,
    'chroma_mean': chroma_means,
    'chroma_var': chroma_vars,
    'spectral_contrast_mean': spectral_contrast_means,
    'zcr_mean': audio_features_df['zcr_mean'],
    'rms_mean': audio_features_df['rms_mean'],
    'tempo': audio_features_df['tempo'],
    'duration': audio_features_df['duration']
})

df = pd.read_csv('audiofile_dataset.csv')

df_with_merged_features = pd.concat([df, merged_features_df], axis=1)

df_with_merged_features.to_csv('audiofile_dataset_with_merged_features.csv', index=False)

print("Dataset with merged audio features saved as 'audiofile_dataset_with_merged_features.csv'")
