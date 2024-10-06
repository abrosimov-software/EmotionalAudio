import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file_path = 'audiofile_dataset_with_wavelet_features.csv'
df = pd.read_csv(csv_file_path)

output_directory = 'Wavelets'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for emotion in df['emotion'].unique():
    emotion_data = df[df['emotion'] == emotion]

    wavelet_coeffs_mean = emotion_data.filter(like='wavelet_coeff_').filter(like='_mean').values
    wavelet_coeffs_var = emotion_data.filter(like='wavelet_coeff_').filter(like='_var').values

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(wavelet_coeffs_mean.T, aspect='auto', cmap='jet')
    plt.colorbar(label='Mean Coefficient Value')
    plt.title(f'Wavelet Coefficient Means - Emotion: {emotion}')
    plt.xlabel('Audio Samples')
    plt.ylabel('Wavelet Coefficient Index')

    plt.subplot(2, 1, 2)
    plt.imshow(wavelet_coeffs_var.T, aspect='auto', cmap='jet')
    plt.colorbar(label='Variance Coefficient Value')
    plt.title(f'Wavelet Coefficient Variances - Emotion: {emotion}')
    plt.xlabel('Audio Samples')
    plt.ylabel('Wavelet Coefficient Index')

    plt.tight_layout()

    emotion_directory = os.path.join(output_directory, emotion)
    if not os.path.exists(emotion_directory):
        os.makedirs(emotion_directory)

    plt.savefig(os.path.join(emotion_directory, f'{emotion}_wavelet_coefficients.png'))
    plt.close()
