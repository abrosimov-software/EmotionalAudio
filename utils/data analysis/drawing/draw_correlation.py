import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

def clean_tempo(tempo_value):
    if isinstance(tempo_value, str):
        return float(tempo_value.strip('[]'))
    return tempo_value

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, '../../../data/raw/audiofile_dataset_with_merged_features.csv'))

df['tempo'] = df['tempo'].apply(clean_tempo)

df = df.drop(columns=['id'])

categorical_columns = ['vocal_channel', 'gender']

label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

encoder = OneHotEncoder(sparse_output=False)

encoded_features = encoder.fit_transform(df[['emotion', 'emotional_intensity']])

encoded_columns = encoder.get_feature_names_out(['emotion', 'emotional_intensity'])

encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)

df_encoded = pd.concat([df.drop(columns=['emotion', 'emotional_intensity']), encoded_df], axis=1)

post_emotion_features = list(encoded_columns)

other_features = [col for col in df_encoded.columns if col not in post_emotion_features]

os.makedirs(os.path.join(current_dir, f'../../../reports/plots/heatmaps', exist_ok=True))

for post_emotion_feature in post_emotion_features:
    correlation_data = df_encoded[[post_emotion_feature] + other_features].corr()
    
    post_emotion_corr = correlation_data[[post_emotion_feature]].drop(index=post_emotion_feature)
    
    post_emotion_corr = post_emotion_corr.T
    
    plt.figure(figsize=(12, 4))
    sns.heatmap(post_emotion_corr, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)
    
    plt.title(f"Correlation of {post_emotion_feature} with other features")
    plt.xlabel("Correlation")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.3, right=0.9, top=0.9)
    
    plt.savefig(os.path.join(current_dir, f'../../../reports/plots/heatmaps/heatmap_{post_emotion_feature}.png'))
    
    plt.close()


os.makedirs(os.path.join(current_dir, f'../../../reports/plots/histograms', exist_ok=True))

for post_emotion_feature in post_emotion_features:
    plt.figure()
    plt.hist(encoded_df[post_emotion_feature], bins=2, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {post_emotion_feature}')
    plt.xlabel(post_emotion_feature)
    plt.ylabel('Count')

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.3, right=0.9, top=0.9)

    plt.savefig(os.path.join(current_dir, f'../../../reports/plots/histograms/histogram_{post_emotion_feature}.png'), bbox_inches='tight', dpi=300)

    plt.close()
