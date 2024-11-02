import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetWithFeatures(nn.Module):
    """
    ResNet model.
    
    Uses pretrained ResNet18 for spectrograms, 
    final classification on linear layer with additional features.
    """
    def __init__(self, model_cfg, output_size):
        super(ResNetWithFeatures, self).__init__()

        self.resnet = resnet18(pretrained=True)
        
        self.resnet.conv1 = nn.Conv2d(**model_cfg["conv"])
        
        num_resnet_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity() 

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc_features = nn.Linear(model_cfg["additional_features_size"], num_resnet_features)
        
        self.fc_final = nn.Linear(num_resnet_features * 2, output_size)

    def forward(self, spectrogram_batch, additional_features_tensor):

        x = self.resnet(spectrogram_batch)

        additional_features_proj = F.relu(self.fc_features(additional_features_tensor))

        combined = torch.cat((x, additional_features_proj), dim=1)

        return self.fc_final(combined)



# ##
# spectrogram_batch = ... #audio tensor
# additional_features_tensor = ... #features tensor
# labels_tensor = ... #labels_tensor

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet18
# from torchvision import transforms
# import torch.optim as optim

# class ResNetWithFeatures(nn.Module):
#     def __init__(self, num_classes, additional_features_size):
#         super(ResNetWithFeatures, self).__init__()
        
#         self.resnet = resnet18(pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         num_resnet_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Identity()  

#         for param in self.resnet.parameters():
#             param.requires_grad = False

#         self.fc_features = nn.Linear(additional_features_size, num_resnet_features)
#         self.fc_final = nn.Linear(num_resnet_features * 2, num_classes)

#     def forward(self, spectrogram_batch, additional_features_tensor):
#         x = self.resnet(spectrogram_batch)
#         additional_features_proj = F.relu(self.fc_features(additional_features_tensor))
#         combined = torch.cat((x, additional_features_proj), dim=1)
#         return self.fc_final(combined)

# num_classes = 9
# additional_features_size = additional_features_tensor.shape[1]  # Replace with actual feature size

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNetWithFeatures(num_classes=num_classes, additional_features_size=additional_features_size).to(device)

# inp_stacked_batch = spectrogram_batch.to(device)
# inp_stacked_batch = inp_stacked_batch.unsqueeze(1) if inp_stacked_batch.dim() == 3 else inp_stacked_batch
# normalize = transforms.Normalize(mean=[0.5], std=[0.5])
# inp_stacked_batch = normalize(inp_stacked_batch)
# additional_features_tensor = additional_features_tensor.to(device)
# labels_tensor = labels_tensor.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)  # Increased learning rate

# num_epochs = 30
# for epoch in range(num_epochs):
#     model.train()
    
#     outputs = model(inp_stacked_batch, additional_features_tensor)
#     loss = criterion(outputs, labels_tensor)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

#     if epoch == 10:
#         for param in model.resnet.parameters():
#             param.requires_grad = True

# model.eval()
# with torch.no_grad():
#     predictions = model(inp_stacked_batch, additional_features_tensor)
#     predicted_classes = torch.argmax(predictions, dim=1)
#     accuracy = (predicted_classes == labels_tensor).float().mean().item()
#     print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
#     print("Predicted Classes:", predicted_classes.cpu().numpy())