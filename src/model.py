import torch.nn as nn
import torchvision.models as models

class VideoDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoDetector, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove last layer
        
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)  # (batch*seq, 2048)
        features = features.view(batch_size, seq_len, -1)
        _, (h_n, _) = self.lstm(features)
        out = self.fc(h_n.squeeze(0))
        return out