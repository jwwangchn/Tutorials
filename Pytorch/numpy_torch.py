import numpy as np
import torch
from torch.autograd import Variable

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)

torch2array = torch_data.numpy()

print('numpy: ', np_data, '\ntorch: ', torch_data, '\ntorch2array: ', torch2array)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)    # 32bit float point
print('\nnumpy: ', np.abs(data), '\ntorch: ', torch.abs(tensor))
print('\nnumpy: ', np.sin(data), '\ntorch: ', torch.sin(tensor))
print('\nnumpy: ', np.mean(data), '\ntorch: ', torch.mean(tensor))

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
print('\nnumpy: ', np.matmul(data, data), '\ntorch: ', torch.mm(tensor, tensor))

# data = [[1,2],[3,4]]
# tensor = torch.FloatTensor(data)
# data = np.array(data)
# print('\nnumpy: ', data.dot(data), '\ntorch: ', tensor.dot(tensor))