import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        self.img_list = list(dir_path_resolved.glob("*/*.png"))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        return img


if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    print("#1.1")
    print(len(my_dataset))
    print("#1.2")
    print(my_dataset[0].size)
    plt.imshow(my_dataset[0])
    plt.show()
