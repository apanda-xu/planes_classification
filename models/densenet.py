import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = torchvision.models.densenet201(pretrained=False)
        self.features = densenet.features
        self.fcs = nn.Sequential(
            nn.Linear(in_features=1920, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=6, bias=True),
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = F.relu(x1, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)
        x3 = self.fcs(x2)
        return x3
