import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=256, kernel_size=5, stride=8, padding=2
        )
        self.batch = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=256 * 16 * 16, out_features=64, bias=True)
        return

    def forward(self, _in):
        _in = self.conv(_in)
        _in = self.batch(_in)
        _in = self.relu(_in)
        _in = _in.view(32, 256 * 16 * 16)
        _in = self.fc(_in)
        return _in
