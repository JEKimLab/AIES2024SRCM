import torch.nn as nn


class PurchaseClassifier(nn.Module):
    def __init__(self, num_classes=100, droprate=0):
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
        if droprate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(droprate),
                nn.Linear(128, num_classes)
            )
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out), hidden_out
