import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import cifar_datasets
from model import CNN

train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()

criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    train_loss = 0
    val_loss = 0

    model.train()
    for images, labels in train_loader:
        train_outputs = model(images)

        loss = criterion(train_outputs, labels)

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
