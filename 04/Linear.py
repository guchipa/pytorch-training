import torch
from torch import nn

if __name__ == "__main__":
    # 1
    my_tensor = torch.ones((32, 1024))
    print("#1\n", repr(my_tensor.size()))

    # 2
    fc2 = nn.Linear(in_features=1024, out_features=256, bias=True)
    print("#2\n", repr(fc2(my_tensor).size()))

    # 3
    fc3 = nn.Linear(in_features=1024, out_features=2048, bias=True)
    print("#3\n", repr(fc3(my_tensor).size()))

    # omake
    out2 = torch.reshape(fc2(my_tensor), (-1, 16, 16))
    print("#omake\n", repr(out2.size()))
