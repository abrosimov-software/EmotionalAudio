import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.nn import functional as F

##
spectrogram_batch = ... #audio tensor
additional_features_tensor = ... #features tensor
labels_tensor = ... #labels_tensor

class CNNTransformerWithFeatures(nn.Module):
    def __init__(self, num_classes, feature_size):
        super(CNNTransformerWithFeatures, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   
        )

        test_input = torch.zeros(1, 1, 345, 128)
        with torch.no_grad():
            cnn_output = self.cnn(test_input)
            _, channels, height, width = cnn_output.shape
            self.flattened_dim = channels * height 

        transformer_layer = TransformerEncoderLayer(d_model=self.flattened_dim, nhead=8)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=6)

        self.fc_features = nn.Linear(feature_size, self.flattened_dim)
        
        self.fc = nn.Linear(self.flattened_dim * 2, num_classes)

    def forward(self, spectrogram_batch, additional_features_tensor):
        x = self.cnn(spectrogram_batch)
        batch_size, channels, height, width = x.shape

        x = x.view(batch_size, channels * height, width).permute(2, 0, 1) 

        x = self.transformer(x)
        x = x.mean(dim=0)  

        additional_features_proj = torch.relu(self.fc_features(additional_features_tensor))

        combined = torch.cat((x, additional_features_proj), dim=1)

        return self.fc(combined)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inp_stacked_batch = spectrogram_batch.to(device)  
# inp_stacked_batch = inp_stacked_batch.unsqueeze(1) if inp_stacked_batch.dim() == 3 else inp_stacked_batch

# num_classes = int(labels_tensor.max().item()) + 1  
# feature_size = additional_features_tensor.shape[1]  

# model = CNNTransformerWithFeatures(num_classes=num_classes, feature_size=feature_size)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# additional_features_tensor = additional_features_tensor.to(device)
# labels_tensor = labels_tensor.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 1  
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
    
