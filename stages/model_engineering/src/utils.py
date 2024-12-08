import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchaudio
from typing import Tuple
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, data_dir, metadata_file, sample_rate=16000):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_file)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_name = row["file_name"]
        target = row["target"]

        audio_path = os.path.join(self.data_dir, "audiofiles", file_name)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found.")

        waveform, orig_sample_rate = torchaudio.load(audio_path)

        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate)(waveform)

        return waveform.squeeze(0).numpy(), target

class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        audios, labels = zip(*batch)

        inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)

        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.get("attention_mask", None),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def get_datasets(
        data_dir: str,
        metadata_file: str,
        sample_rate: int = 16000,
        random_seed: int = 42,
        train_split: float = 0.8
) -> Tuple[Dataset, Dataset]:
    dataset = AudioDataset(data_dir, metadata_file, sample_rate)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_end = int(train_split * dataset_size)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset



def collate_fn(batch):
    waveforms, targets = zip(*batch)

    waveforms_padded = pad_sequence(waveforms, batch_first=True)

    targets = torch.tensor(targets, dtype=torch.long)

    return {"input_values": waveforms_padded, "labels": targets}

