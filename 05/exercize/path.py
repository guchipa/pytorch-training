from pathlib import Path

data_directory = "./data"

# 1
data_directory_path = Path(data_directory).resolve()
print("#1", data_directory_path)

# 2
file_list = list(data_directory_path.glob("*"))
print("#2", file_list)

# 3
img_list = list(data_directory_path.glob("*/*.png"))
print("#3", len(img_list))
