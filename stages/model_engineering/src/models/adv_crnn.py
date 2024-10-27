
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CRNNModel(nn.Module):
    """
    Custom CRNN model.
    
    Uses CNN to extract features from the spectrogram.
    Uses LSTM to process the CNN output.
    Linear layer is used to bring the output to the desired output size.
    Multiclass classification is done using CrossEntropyLoss.
    """
    def __init__(self, model_cfg, output_size):
        super(CRNNModel, self).__init__()

        self.output_size = output_size

        self.cnn = nn.Sequential(
            nn.Conv1d(**model_cfg["conv1"]),
            nn.ReLU(),
            nn.MaxPool1d(**model_cfg["pool1"]),
            nn.Conv1d(**model_cfg["conv2"]),
            nn.ReLU(),
            nn.MaxPool1d(**model_cfg["pool2"])
        )

        self.lstm = nn.LSTM(**model_cfg["lstm"], batch_first=True)

        self.fc = nn.LazyLinear(output_size)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, input_size, seq_len)

        x = self.cnn(x)
        x = x.permute(0, 2, 1) # Shape: (batch_size, seq_len, num_features)
        x, _ = self.lstm(x)

        out = self.fc(x)

        return out


# ##
# spectrogram_batch = ... #audio tensor
# additional_features_tensor = ... #features tensor
# labels_tensor = ... #labels_tensor


# class CRNNWithFeatures(nn.Module):
#     def __init__(self, input_shape, feature_size, num_classes):
#         super(CRNNWithFeatures, self).__init__()
        
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # Set in_channels to 1 for audio
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         with torch.no_grad():
#             example_input = torch.zeros(1, *input_shape[1:]) 
#             example_input = example_input.unsqueeze(1) if example_input.dim() == 3 else example_input
#             cnn_out = self.cnn(example_input)
#             _, channels, height, width = cnn_out.shape
#             self.lstm_input_size = channels * height 
#             self.time_steps = width 
        
#         self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=256, num_layers=2, batch_first=True)
        
#         self.fc_features = nn.Linear(feature_size, 256)
        
#         self.fc = nn.Linear(256 * 2, num_classes)

#     def forward(self, spectrogram_batch, additional_features_tensor):
#         x = self.cnn(spectrogram_batch)
#         batch_size, channels, height, width = x.shape
#         x = x.view(batch_size, self.time_steps, self.lstm_input_size)  # Shape: (batch_size, time_steps, lstm_input_size)
#         x, _ = self.lstm(x)
#         spectrogram_representation = x[:, -1, :]  
#         additional_features_proj = torch.relu(self.fc_features(additional_features_tensor))
#         combined = torch.cat((spectrogram_representation, additional_features_proj), dim=1)

#         # Final classification layer
#         return self.fc(combined)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# inp_stacked_batch = spectrogram_batch.to(device)  
# inp_stacked_batch = inp_stacked_batch.unsqueeze(1) if inp_stacked_batch.dim() == 3 else inp_stacked_batch
# additional_features_tensor = additional_features_tensor.to(device)
# labels_tensor = labels_tensor.to(device)


# input_shape = inp_stacked_batch.shape 
# feature_size = additional_features_tensor.shape[1]  
# num_classes = 9  

# model = CRNNWithFeatures(input_shape=input_shape, feature_size=feature_size, num_classes=num_classes)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)


# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()  

#     outputs = model(inp_stacked_batch, additional_features_tensor)
#     loss = criterion(outputs, labels_tensor)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# model.eval() 
# with torch.no_grad():
#     predictions = model(inp_stacked_batch, additional_features_tensor)
#     predicted_classes = torch.argmax(predictions, dim=1)
    
#     print("Predicted Classes:", predicted_classes.cpu().numpy()) 