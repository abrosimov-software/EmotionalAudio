import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomEmotionDetector(nn.Module):
    # This is a random classifier for testing purposes
    # Processes input audio and returns a random emotion
    def __init__(self, num_classes=7):
        super(RandomEmotionDetector, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.randint(0, self.num_classes, (x.size(0),))