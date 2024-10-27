import os
import torch
from torch.utils.data import (
    Dataset, 
    # DataLoader
)

class AudioDataset(Dataset):
    def __init__(self, audio_dir, features_dir, labels_dir):
        self.audio_files = sorted(os.listdir(audio_dir))
        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))
        
        self.audio_dir = audio_dir
        self.features_dir = features_dir
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio tensor
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        audio = torch.load(audio_path)
        
        # Load features tensor
        features_path = os.path.join(self.features_dir, self.features_files[idx])
        features = torch.load(features_path)
        
        # Load labels tensor
        labels_path = os.path.join(self.labels_dir, self.labels_files[idx])
        labels = torch.load(labels_path)
        
        return audio, features, labels

# # Set the directories
# audio_dir = '../../data/preprocessed/audiofiles'
# features_dir = '../../data/preprocessed/features'
# labels_dir = '../../data/preprocessed/labels'

# # Instantiate the dataset and dataloader
# dataset = AudioDataset(audio_dir, features_dir, labels_dir)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
