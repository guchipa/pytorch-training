# モジュールのインポート
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# dataset.py内のdatasets関数をインポート
from dataset import cifar_datasets

# model.py内のCNNクラスをインポート
from model import CNN

# データローダーからデータを受け取る
train_data, test_data = cifar_datasets()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# モデル、損失関数、最適化関数の定義
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == "__main__":
    epochs = 20

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)

        # val
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader)
        avg_val_acc = val_acc / len(test_loader.dataset)

        print(
            "Epoch {}, Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}".format(
                epoch + 1,
                i + 1,
                loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_acc=avg_val_acc,
            )
        )
