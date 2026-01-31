import torch
import torch.nn as nn
import torchvision.models as models

class VideoDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoDetector, self).__init__()

        # CNN backbone
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # remove classifier → 2048-d features

        # LSTM with regularization
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=2,          #  ADDED
            dropout=0.5,           #  REDUCED for better learning capacity
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)  # Reduced dropout for better learning
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        b, t, C, H, W = x.size()

        x = x.view(b * t, C, H, W)
        features = self.cnn(x)                 # (b*t, 2048)
        features = features.view(b, t, -1)     # (b, t, 2048)

        lstm_out, _ = self.lstm(features)      # (b, t, 512)

        # TEMPORAL AVERAGE POOLING (reverted from attention)
        video_feat = lstm_out.mean(dim=1)      # (b, 512)

        out = self.dropout(self.fc(video_feat))
        return out
