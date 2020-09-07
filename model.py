import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = torchvision.models.vgg16(pretrained=True).features[:-1]
        # freeze all layers except the last two conv layers
        for i, param in enumerate(self.encoder.parameters()):
            if i == 24: break
            param.requires_grad = False
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 1, 1, padding=0) #, nn.Sigmoid()
        )

        self.classifier = nn.Sigmoid()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.classifier(x)
        return x
            
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d = nn.Sequential( 
            nn.Conv2d(4, 3, 1, padding=1), nn.ReLU(), 
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), 
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.linear = nn.Sequential(
            nn.Linear(64*28*28, 100), nn.Tanh(),
            nn.Linear(100,2), nn.Tanh(),
            nn.Linear(2,1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
                
class BCELossWithDownsampling():
    def __init__(self):
        self.downsample = nn.AvgPool2d(4, stride=4, count_include_pad=False)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def __call__(self, pred, y):
        return self.loss_fn(self.downsample(pred), self.downsample(y))

class WeightedMSELoss():
    def __init__(self):
        self.loss_fn = nn.MSELoss(reduction='none')
        self.alpha = 1.1  # hpyerparameter
        
    def __call__(self, pred, y):
        L = self.loss_fn(pred, y)
        w = 1 / ((self.alpha - y) ** 2)
        return (w * L).mean()