import torch


aa = torch.tensor([0,1,2,3,4,5,6,7,8,9])
print(aa[0:10:2])
print(aa[1:10:2])

bb = [1 ,2,3,4,5,6,7,8,9]
print(bb[0])
print(bb[::2])
print(bb[1::2])

indices = torch.randperm(5100)
print("idx:",torch.where(indices>2000))
label_indices = indices[0::2]

print(indices)

aa = torch.tensor([1,2,3,4,5])
print(aa[aa > 3])

aa = torch.tensor([[10,10,20,20]])
print(aa.new_ones(aa.shape[0]))


print(torch.randperm(100))

aa = torch.tensor([[1,2,3,4,5],
                   [6,7,8,9,10]])
print(aa[:,:-1])


aa = torch.rand((3,2))
bb = torch.rand((2, 5))
print(torch.mm(aa, bb))


aa = torch.zeros(10)
print(aa)
mask = torch.tensor([True, True, False, True, True, False, True, True, False, False])
bb = torch.rand(10)[:6]
print(bb)
print(torch.masked_scatter(aa, mask, bb))

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')
cat2label = {cat: i for i, cat in enumerate(CLASSES)}


aa = torch.rand((2, 3, 4))
print(aa)
print(aa.permute(1, 0, 2).reshape(3, -1))

aa = torch.rand((100, 10))
print(torch.softmax(aa, dim=1))

# indx = torch.arange(len(4))
indy = torch.tensor([3,1,2,3])
aa = torch.rand((4, 6))
print(aa)
print(aa[:, indy].diag())

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def func(x):
    y = tanh((x)*3)
    return y

# x = np.arange(0, 1, 0.01)
# # y = 1. / np.exp(1-x)
# y = func(x)
# plt.plot(x, y)
# plt.plot(x, x)
# plt.show()


aa = torch.rand((1, 128))

bb = torch.rand((128, 5))
cc = torch.rand((128, 0))
print(torch.clamp(torch.mm(aa.reshape(1, -1), cc), min=0.).sum() / (cc.shape[1] +1))
print(torch.mm(aa, bb))
print(torch.mm(aa, cc))

# print(torch.mm(aa.cuda(), bb.cuda()))
# dict(
#     type='Compose',
#     transforms=[
#         dict(type='RandErasing', p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
#         dict(type='RandErasing', p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
#         dict(type='RandErasing', p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
#
#     ]
# ),