import torch
from torch import nn
from models.mymodel import MyModel


if __name__ == "__main__":
    in_tensor = torch.ones((32, 3, 128, 128))

    model = MyModel()

    out = model(in_tensor)
    print(repr(out.size()))
