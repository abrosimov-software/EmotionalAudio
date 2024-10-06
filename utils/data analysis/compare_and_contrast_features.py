import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import os

def clean_tempo(tempo_value):
    if isinstance(tempo_value, str):
        return float(tempo_value.strip('[]'))
    return tempo_value

df = pd.read_csv('audiofile_dataset_with_merged_features.csv')

df['tempo'] = df['tempo'].apply(clean_tempo)

df = df.drop(columns=['id'])

categorical_columns = ['vocal_channel', 'gender', 'emotional_intensity']

label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

unique_emotions = df['emotion'].unique()

def compare_all_emotions(df, emotion_list, feature_columns):
    for emotion1, emotion2 in combinations(emotion_list, 2):
        print(f"Comparing {emotion1} and {emotion2}\n")
        
        emotion1_df = df[df['emotion'] == emotion1]
        emotion2_df = df[df['emotion'] == emotion2]
        
        summary_df1 = emotion1_df[feature_columns].describe()
        summary_df2 = emotion2_df[feature_columns].describe()
        
        if not os.path.exists(f'Comparisons/{emotion1} vs {emotion2}'):
            os.makedirs(f'Comparisons/{emotion1} vs {emotion2}')

        for feature in feature_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(emotion1_df[feature], color='blue', label=emotion1, kde=True, stat='density', bins=30, alpha=0.5)
            sns.histplot(emotion2_df[feature], color='orange', label=emotion2, kde=True, stat='density', bins=30, alpha=0.5)
            plt.title(f'Histogram Comparison of {emotion1} and {emotion2} for {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()

            plt.savefig(f'Comparisons/{emotion1} vs {emotion2}/comparison_{emotion1}_vs_{emotion2}_{feature}.png', bbox_inches='tight', dpi=300)
            plt.close()

feature_columns = [col for col in df.columns if col not in ['id', 'emotion']]

compare_all_emotions(df, unique_emotions, feature_columns)

print("Comparison plots saved as .png files.")
