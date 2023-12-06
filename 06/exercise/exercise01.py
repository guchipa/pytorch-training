from cv2 import transform
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    image_path = "./exercise_data/dog_img.png"
    image = Image.open(image_path)

    # 1
    preprocess_1 = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    processed_image_1 = preprocess_1(image)
    plt.imshow(processed_image_1.permute((1, 2, 0)))
    plt.show()

    # 2
    preprocess_2 = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.55),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
        ]
    )

    processed_image_2 = preprocess_2(image)
    plt.imshow(processed_image_2.permute((1, 2, 0)))
    plt.show()

    # 1
    preprocess_3 = transforms.Compose(
        [
            transforms.RandomRotation(degrees=100),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0)),
            transforms.ToTensor(),
        ]
    )

    processed_image_3 = preprocess_3(image)
    plt.imshow(processed_image_3.permute((1, 2, 0)))
    plt.show()
