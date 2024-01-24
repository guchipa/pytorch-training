import torch
from torch.utils.data import DataLoader
from dataset import get_cifar_datasets
from model import CNN
from torchvision.models import resnet18

mymodel_path = "cifar_cnn.pth"
resnet18_path = "resnet18_cnn.pth"

_, test_data = get_cifar_datasets()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

mymodel = CNN()
resnet_model = resnet18()

mymodel.load_state_dict(torch.load(mymodel_path)["model_state_dict"])
mymodel.eval()

resnet_model.load_state_dict(torch.load(resnet18_path)["model_state_dict"])
resnet_model.eval()


if __name__ == "__main__":
    val_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = mymodel(images)
            val_acc += (outputs.max(1)[1] == labels).sum().item()

    avg_val_acc = val_acc / len(test_loader.dataset)

    print("---mymodel---")
    print("Accuracy: {:.4f}".format(avg_val_acc))

    val_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = resnet_model(images)
            val_acc += (outputs.max(1)[1] == labels).sum().item()

    avg_val_acc = val_acc / len(test_loader.dataset)

    print("---ResNet18---")
    print("Accuracy: {:.4f}".format(avg_val_acc))
