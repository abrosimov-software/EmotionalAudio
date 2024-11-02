import os
import torch
import numpy as np
from torch.utils.data import (
    Dataset, 
    DataLoader,
    SubsetRandomSampler
)
import torchaudio

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
    

class AnotherAudioDataset(Dataset):
    def __init__(self, audio_dir, target_sample_rate=16000):
        self.audio_files = sorted(os.listdir(audio_dir))
        self.identifiers = [list(map(int, audio_file[:-4].split('-'))) for audio_file in self.audio_files]
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_classes = len(set(identifier[2] for identifier in self.identifiers)) + 1 # Add 1 for unknown class

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        audio, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio = resampler(audio)
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio, torch.tensor(self.identifiers[idx][2]), torch.tensor(self.identifiers[idx])

# Collate function for padding and batching
def collate_audio(batch):
    audios, targets, additional_features = zip(*batch)
    
    # Find the maximum length in the batch
    max_length = max(audio.shape[1] for audio in audios)
    
    # Pad each audio to the max length
    padded_audios = [torch.nn.functional.pad(audio, (0, max_length - audio.shape[1])) for audio in audios]
    padded_audios = torch.stack(padded_audios)
    
    return padded_audios, torch.stack(targets), torch.stack(additional_features)

# Function to split dataset by identifiers
def create_data_loaders(dataset, batch_size=4, train_ratio=0.8, shuffle=True):
    # Extract actor IDs from identifiers for splitting
    actor_ids = [identifier[5] for identifier in dataset.identifiers]
    unique_actors = np.unique(actor_ids)
    
    # Shuffle actors to randomize train/test split
    if shuffle:
        np.random.shuffle(unique_actors)
    
    # Split actors into training and testing
    split_idx = int(len(unique_actors) * train_ratio)
    train_actors = unique_actors[:split_idx]
    test_actors = unique_actors[split_idx:]
    
    # Create index lists based on actor IDs
    train_indices = [i for i, actor_id in enumerate(actor_ids) if actor_id in train_actors]
    test_indices = [i for i, actor_id in enumerate(actor_ids) if actor_id in test_actors]
    
    # Define samplers for train and test loaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_audio)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_audio)
    
    return train_loader, test_loader


# Calculate loss for training
def calculate_loss(output, targets):
    # Output shape: (n_batch, length, n_classes)
    # Targets shape: (n_batch,)
    # Average class probabilities along the length dimension
    avg_output = torch.mean(output, dim=1)  # Shape: (n_batch, n_classes)
    
    # Calculate loss
    loss = torch.nn.functional.cross_entropy(avg_output, targets)
    return loss

# Training function
def train_model(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        audios, targets, _ = batch
        audios, targets = audios.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(audios)  # Output shape: (n_batch, length, n_classes)
        
        # Calculate loss
        loss = calculate_loss(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            audios, targets, _ = batch
            audios, targets = audios.to(device), targets.to(device)
            
            # Forward pass
            output = model(audios)

            # Calculate loss
            loss = calculate_loss(output, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Testing Loss: {avg_loss:.4f}")
    return avg_loss
        

# # Set the directories
# audio_dir = '../../data/preprocessed/audiofiles'
# features_dir = '../../data/preprocessed/features'
# labels_dir = '../../data/preprocessed/labels'

# # Instantiate the dataset and dataloader
# dataset = AudioDataset(audio_dir, features_dir, labels_dir)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
