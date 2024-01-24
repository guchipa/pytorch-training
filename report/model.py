import torch.nn as nn

NUM_CLASSES = 10


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(64 * 16 * 16, 512)
        self.l2 = nn.Linear(512, 64)
        self.l3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.pool,
        )

        self.classifier = nn.Sequential(
            self.l1,
            self.l2,
            self.l3,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  # 形状の調整
        x = self.classifier(x)
        return x
