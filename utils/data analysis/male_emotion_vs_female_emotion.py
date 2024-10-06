import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('audiofile_dataset_with_merged_features.csv')

emotions = df['emotion'].unique()

features_to_compare = df.columns.difference(['id', 'emotion', 'emotional_intensity', 'vocal_channel', 'gender'])

for emotion in emotions:
    for feature in features_to_compare:
        male_entries = df[(df['emotion'] == emotion) & (df['gender'] == 'male')]
        female_entries = df[(df['emotion'] == emotion) & (df['gender'] == 'female')]

        if not male_entries.empty and not female_entries.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(male_entries[feature], bins=30, alpha=0.5, label='Male', color='blue')
            plt.hist(female_entries[feature], bins=30, alpha=0.5, label='Female', color='red')
            plt.title(f'Distribution of {feature} for Male and Female with Emotion: {emotion}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()

            output_dir = f'Comparisons/Male_Female_Emotions/{emotion}'
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(os.path.join(output_dir, f'{feature}.png'))
            plt.close()

