import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        self.img_list = list(dir_path_resolved.glob("*/*.png"))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        img_tensor = self.transform(img)

        img_path = Path(img_path)
        parts = img_path.parts
        label = parts[9]

        return img_tensor, label


if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    img_tensor, label = my_dataset[45]

    print("#1.1")
    print(img_tensor.size())
    print("#1.2")
    print(label)
    # plt.imshow(my_dataset[0])
    # plt.show()
