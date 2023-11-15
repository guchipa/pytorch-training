from torch import nn
import torch


class ExcersizeModel(nn.Module):
    def __init__(self, mytensor, elem_add, elem_multiply):
        super().__init__()
        self.mytensor = mytensor
        self.elem_add = elem_add
        self.elem_multiply = elem_multiply

    def forward(self, x):
        p2 = x + self.mytensor
        p3 = p2 + self.elem_add
        p4 = p3 * self.elem_multiply
        return p2, p3, p4


if __name__ == "__main__":
    mymodel = ExcersizeModel(torch.ones((3, 3)), 4, 6)

    p2out, p3out, p4out = mymodel(torch.full((3, 3), 2))

    print("# 2")
    print(repr(p2out))
    print("# 3")
    print(repr(p3out))
    print("# 4")
    print(repr(p4out))
