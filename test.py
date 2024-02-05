import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

pos_mat = torch.arange(5).reshape((-1, 1))
print(pos_mat)

#
# i_mat = torch.arange(0, 8, 2).reshape(1, -1)
#
# print(i_mat)
#
# # 创建一个一维数组
# arr = np.arange(12).reshape((2, -1))
#
# print(arr)
