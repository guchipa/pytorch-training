import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_cifar_datasets
from model import CNN
from torchvision.models import resnet18


def cifar_training(model_path, epochs):
    train_data, _ = get_cifar_datasets()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("Training with my model...")

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

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

        # モデルの保存
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
            },
            model_path,
        )

        print(
            "Epoch: {}, Loss: {loss:4f}, Acc: {acc:4f}".format(
                epoch + 1, i + 1, loss=avg_train_loss, acc=avg_train_acc
            )
        )


def resnet_training(model_path, epochs):
    train_data, _ = get_cifar_datasets()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("Training with ResNet18...")

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

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

        # モデルの保存
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
            },
            model_path,
        )

        print(
            "Epoch: {}, Loss: {loss:4f}, Acc: {acc:4f}".format(
                epoch + 1, i + 1, loss=avg_train_loss, acc=avg_train_acc
            )
        )


if __name__ == "__main__":
    # cifar_training(model_path="cifar_cnn.pth", epochs=20)
    resnet_training(model_path="resnet18_cnn.pth", epochs=20)
