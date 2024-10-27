from src.utils import *
from src.models.base_lstm_mpool import LSTMModel
import torch
import mlflow
import mlflow.pytorch
import os
import yaml

# Load model config from /app/config/models.yaml
model_cfg = yaml.safe_load(open("/app/config/models.yaml"))

# Load dataset
data_path = "/app/data/training_data"
dataset = AnotherAudioDataset(data_path)

# Test the dataset
for audio, emotion, identifiers in dataset:
    print(audio.shape, emotion, identifiers)
    break

# Create data loaders
train_loader, test_loader = create_data_loaders(dataset, batch_size=8)

# Test the data loaders
for audio, emotion, identifiers in train_loader:
    print(audio.shape, emotion, identifiers)
    break

# Define model
model = LSTMModel(model_cfg["simple_lstm"], dataset.num_classes)

# Test the model output on a batch
for audio, emotion, identifiers in train_loader:
    output = model(audio)
    print(output.shape, calculate_loss(output, emotion))
    break

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for epoch in range(10):
    print(train_model(model, train_loader, optimizer, device))

    

