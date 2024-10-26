import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def clean_tempo(tempo_value):
    if isinstance(tempo_value, str):
        return float(tempo_value.strip('[]'))
    return tempo_value

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, '../../../data/raw/audiofile_dataset_with_merged_features.csv'))

df['tempo'] = df['tempo'].apply(clean_tempo)

df = df.drop(columns=['id', 'emotion', 'emotional_intensity', 'vocal_channel'])

genders = df['gender'].unique()

def compare_genders(df, emotion_list, feature_columns):
    female = df[df['gender'] == "female"]
    male = df[df['gender'] == "male"]
    
    os.makedirs(os.path.join(current_dir, f'../../../reports/plots/Comparisons/male vs female'), exist_ok=True)

    for feature in feature_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(female[feature], color='blue', label="female", kde=True, stat='density', bins=30, alpha=0.5)
        sns.histplot(male[feature], color='orange', label="male", kde=True, stat='density', bins=30, alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(current_dir, f'../../../reports/plots/Comparisons/male vs female/comparison_{feature}.png'), bbox_inches='tight', dpi=300)
        plt.close()

feature_columns = [col for col in df.columns if col not in ['gender']]

compare_genders(df, genders, feature_columns)
