import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


##
spectrogram_batch = ... #audio tensor
additional_features_tensor = ... #features tensor
labels_tensor = ... #labels_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        
        if self.downsample is not None:
            x = self.downsample(x)            
        out = F.relu(out + x)
        return out

class TCNWithFeatures(nn.Module):
    def __init__(self, input_height, feature_size, num_classes, num_levels=4, kernel_size=3):
        super(TCNWithFeatures, self).__init__()
        
        layers = []
        num_channels = [input_height] + [64] * num_levels
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers += [TemporalBlock(num_channels[i], num_channels[i + 1], kernel_size, stride=1, dilation=dilation_size)]
        self.tcn = nn.Sequential(*layers)
        
        self.fc_features = nn.Linear(feature_size, 64)
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, spectrogram_batch, additional_features_tensor):
        batch_size, _, height, width = spectrogram_batch.shape
        x = spectrogram_batch.view(batch_size, height, width).permute(0, 2, 1)
        
        x = self.tcn(x)
        x = x.mean(dim=2)
        
        additional_features_proj = F.relu(self.fc_features(additional_features_tensor))
        
        combined = torch.cat((x, additional_features_proj), dim=1)
        
        output = self.fc(combined)
        return output

# input_height = spectrogram_batch.shape[2]
# num_classes = labels_tensor.max().item() + 1
# feature_size = additional_features_tensor.shape[1]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inp_stacked_batch = spectrogram_batch.to(device)  
# inp_stacked_batch = inp_stacked_batch.unsqueeze(1) if inp_stacked_batch.dim() == 3 else inp_stacked_batch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = TCNWithFeatures(input_height=input_height, feature_size=feature_size, num_classes=num_classes).to(device)

# additional_features_tensor = additional_features_tensor.to(device)
# labels_tensor = labels_tensor.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 20 

# for epoch in range(num_epochs):
#     model.train()
    
#     outputs = model(inp_stacked_batch, additional_features_tensor)
#     loss = criterion(outputs, labels_tensor)
 
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# def evaluate(model, inp_stacked_batch, additional_features_tensor, labels_tensor, criterion):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(inp_stacked_batch, additional_features_tensor)
#         loss = criterion(outputs, labels_tensor)
#         predicted_classes = torch.argmax(outputs, dim=1)

#         print("Predicted Classes:", predicted_classes.cpu().numpy()) 
#         print(f"Evaluation Loss: {loss.item():.4f}")


# evaluate(model, inp_stacked_batch, additional_features_tensor, labels_tensor, criterion)