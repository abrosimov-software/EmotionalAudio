import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('audiofile_dataset.csv')

df = df.drop(columns=['id'])

features = df.columns

df[features] = df[features].astype(str)

num_features = len(features)
fig, axes = plt.subplots(num_features, num_features, figsize=(18, 18))

for i, feature_x in enumerate(features):
    for j, feature_y in enumerate(features):
        ax = axes[i, j]
        
        if i == j:
            sns.histplot(df[feature_x], ax=ax, kde=False)
            ax.set_ylabel(feature_x)
            ax.set_xlabel(feature_x)
        else:
            sns.histplot(data=df, x=feature_x, hue=feature_y, multiple="stack", ax=ax, kde=False, legend=False)
            ax.set_ylabel('')
            ax.set_xlabel('')
        
        ax.set_title(f'{feature_x} on {feature_y}', fontsize=8)

        ax.tick_params(left=False, bottom=False)

plt.tight_layout()
plt.show()
