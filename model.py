import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.encoder = base.features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 8 → 16
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 2, stride=2),   # 16 → 32
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, stride=2),   # 32 → 64
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),    # 64 → 128
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),     # 128 → 256
            nn.ReLU(),
        )

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(self.final(x))
