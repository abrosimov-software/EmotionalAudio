import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('audiofile_dataset_with_merged_features.csv')

sns.set(style="whitegrid")

def plot_histograms(dataframe):
    plt.figure(figsize=(15, 10))
    
    for i, column in enumerate(dataframe.columns):
        plt.subplot(4, 4, i + 1)
        sns.histplot(dataframe[column], bins=30, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_histograms(df)
