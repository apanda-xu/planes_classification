import torch.nn as nn 
import torchvision

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg13_bn(pretrained=False)
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fcs = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=6, bias=True),
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.avgpool(x1)
        x3 = x2.view(x.shape[0], -1)
        x4 = self.fcs(x3)
        return x4
