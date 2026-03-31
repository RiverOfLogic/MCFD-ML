import torch
import torch.nn as nn
from MEDGNet import FeatureEncoder

class Decoder(nn.Module):
    def __init__(self, in_dim=256, out_channel=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128 * 128),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # -> 256
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), # -> 512
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), # -> 1024
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, out_channel, kernel_size=4, stride=2, padding=1) # -> 2048
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 128)
        x = self.deconv(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_dim=128, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class CDDGNet(nn.Module):
    def __init__(self, in_channels=8, feat_dim=128, num_classes=7):
        super().__init__()
        # Causal Encoder
        self.E_c = FeatureEncoder(input_channel=in_channels, feature_dim=feat_dim)
        # Domain Encoder
        self.E_d = FeatureEncoder(input_channel=in_channels, feature_dim=feat_dim)
        
        self.Decoder = Decoder(in_dim=feat_dim * 2, out_channel=in_channels)
        self.Classifier = Classifier(in_dim=feat_dim, num_classes=num_classes)
        
    def forward(self, x):
        z_c = self.E_c(x)
        z_d = self.E_d(x)
        
        z_concat = torch.cat([z_c, z_d], dim=1)
        x_recon = self.Decoder(z_concat)
        
        y_pred = self.Classifier(z_c)
        
        return y_pred, x_recon, z_c, z_d
