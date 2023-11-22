import torch
from torch import nn

if __name__ == "__main__":
    # 1
    my_tensor = torch.ones((32, 3, 128, 128))
    print("#1\n", my_tensor.size())

    # 2
    conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    out2 = conv2(my_tensor)
    print("#2\n", out2.size())

    # 3
    conv3 = nn.Conv2d(
        in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1
    )
    out3 = conv3(my_tensor)
    print("#3\n", out3.size())

    # 4
    conv4_2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1)
    conv4_3 = nn.Conv2d(
        in_channels=3, out_channels=256, kernel_size=5, padding=2, stride=2
    )
    out4_2 = conv4_2(my_tensor)
    out4_3 = conv4_3(my_tensor)
    print("#4\n", out4_2.size(), "\n", out4_3.size())
