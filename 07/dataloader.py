from torch.utils.data import DataLoader
from dataset import cifar_datasets

train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
