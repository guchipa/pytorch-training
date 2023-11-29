import numpy as np
import torch

if __name__ == "__main__":

  data = np.array(
    [
      [[85, 78], [67, 82], [92, 88], [75, 70], [60, 64]],
      [[70, 68], [77, 72], [85, 90], [60, 65], [78, 76]],
      [[80, 84], [88, 87], [66, 68], [72, 73], [64, 60]]
    ]
  )


  # 1
  data = torch.tensor(data, dtype=float)
  print("# 1\n", torch.Tensor.size(data))
  print(data)

  # 2
  data = torch.permute(data, (2, 0, 1))
  print("# 2\n", torch.Tensor.size(data))

  # 3
  sum_data = torch.sum(data, 0)
  print("# 3\n", torch.sum(data, 0))
  print(torch.Tensor.size(sum_data))

  # 4
  print("# 4\n", torch.mean(sum_data, 1))

  # 5
  print("# 5\n", torch.sum(sum_data, 1) / sum_data.size(1))
