import torch
from torch.utils.data import DataLoader
from dataset import cifar_datasets
from model import CNN

train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = CNN()

train_outputs_list = []
test_outputs_list = []

for epoch in range(1):
    model.train()
    for images, kabeks in train_loader:
        train_outputs = model(images)
        train_outputs_list.append(train_outputs)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            test_outputs = model(images)
            test_outputs_list.append(test_outputs)
