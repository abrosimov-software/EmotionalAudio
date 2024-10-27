import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    """
    Simple LSTM model.
    
    Uses nn.LSTM to process the audio input.
    Linear layer is used to bring the output to the desired output size.
    Multiclass classification is done using CrossEntropyLoss.
    """
    def __init__(self, model_cfg, output_size):
        super(LSTMModel, self).__init__()

        self.output_size = output_size

        self.lstm = nn.LSTM(**model_cfg["lstm"], batch_first=True)
        
        self.fc = nn.LazyLinear(output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))  # (batch, seq, feature) -> (batch, feature, seq)

        out = self.fc(lstm_out)

        return out

##
# spectrogram_batch = ... #audio tensor
# additional_features_tensor = ... #features tensor
# labels_tensor = ... #labels_tensor


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.maxpool = nn.MaxPool1d(2)
#         self.hidden_size = hidden_size

#         self.fc1 = None
#         self.fc2 = nn.Linear(hidden_size, output_size)  
#     def forward(self, x, additional_features):
#         lstm_out, _ = self.lstm(x) 
        
#         lstm_out = lstm_out.permute(0, 2, 1)  
#         pooled_out = self.maxpool(lstm_out).mean(dim=2) 

#         combined_features = torch.cat((pooled_out, additional_features), dim=1)

#         if self.fc1 is None:
#             fc1_input_size = combined_features.shape[1] 
#             self.fc1 = nn.Linear(fc1_input_size, self.hidden_size).to(x.device)

#         out = torch.relu(self.fc1(combined_features))
#         out = self.fc2(out)
#         return out

# input_size = spectrogram_batch.shape[2] 
# hidden_size = 32 
# output_size = 9  

# model = LSTMModel(input_size, hidden_size, output_size)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = model.to(device)

# inp_stacked_batch = spectrogram_batch.to(device)  
# additional_features_tensor = additional_features_tensor.to(device)
# labels_tensor = labels_tensor.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train() 

#     outputs = model(inp_stacked_batch, additional_features_tensor)
#     loss = criterion(outputs, labels_tensor)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# model.eval()  
# with torch.no_grad(): 
#     predictions = model(inp_stacked_batch, additional_features_tensor)
#     predicted_classes = torch.argmax(predictions, dim=1)
#     print("Predicted Classes:", predicted_classes.cpu().numpy()) 