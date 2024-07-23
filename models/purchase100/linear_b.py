import torch.nn as nn
from models.utils.srcm_b import SRCMB

class PurchaseClassifier(nn.Module):
    def __init__(self, num_classes=100, droprate=0, r=15, d=1):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.classifier = SRCMB(128, out_features=num_classes, pre_layer=True, r=r, d=d)

    def forward(self, x):
        out, hidden_out = self.classifier(self.features(x))
        return out, hidden_out
