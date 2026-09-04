import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        return self.network(x)