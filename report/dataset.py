from torchvision import transforms, datasets


def get_cifar_datasets():
    train_data = datasets.CIFAR10(
        root="../", train=True, transform=transforms.ToTensor(), download=True
    )
    test_data = datasets.CIFAR10(
        root="../", train=False, transform=transforms.ToTensor(), download=True
    )

    return train_data, test_data
