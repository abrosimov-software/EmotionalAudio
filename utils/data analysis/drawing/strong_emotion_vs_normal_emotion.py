import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, '../../../data/raw/audiofile_dataset_with_merged_features.csv'))

emotions = df['emotion'].unique()

features_to_compare = df.columns.difference(['id', 'emotion', 'emotional_intensity', 'vocal_channel', 'gender'])

for emotion in emotions:
    for feature in features_to_compare:
        strong_entries = df[(df['emotion'] == emotion) & (df['emotional_intensity'] == 'strong')]
        normal_entries = df[(df['emotion'] == emotion) & (df['emotional_intensity'] == 'normal')]

        if not strong_entries.empty and not normal_entries.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(strong_entries[feature], bins=30, alpha=0.5, label='Strong', color='blue')
            plt.hist(normal_entries[feature], bins=30, alpha=0.5, label='Normal', color='red')
            plt.title(f'Distribution of {feature} for Strong and Normal {emotion}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            
            output_dir = os.path.join(current_dir, f'../../../reports/plots/Comparisons/Intensity_of_Emotions/{emotion}')
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(os.path.join(output_dir, f'{feature}.png'))
            plt.close()

