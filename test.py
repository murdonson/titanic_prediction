import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
m = 400  # 样本数量
N = int(m / 2)  # 每一类的点的个数
D = 2  # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8')  # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N * j, N * (j + 1))
    t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
    r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
    x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# w = nn.Parameter(torch.randn(2, 1))
# b = nn.Parameter(torch.zeros(1))
#
# optimizer = torch.optim.SGD([w, b], 1e-1)
#
#
# def logistic_regression(x):
#     return F.sigmoid(torch.mm(x, w) + b)
#
#
# criterion = nn.BCEWithLogitsLoss()
# for e in range(100):
#     out = logistic_regression(Variable(x))
#     loss = criterion(out, Variable(y))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (e + 1) % 20 == 0:
#         print('epoch:{},loss:{}'.format(e + 1, loss.data[0]))
#

# w1=nn.Parameter(torch.randn(2,4)*0.01)
# b1=nn.Parameter(torch.zeros(4))

seq_net = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

param = seq_net.parameters()
optim = torch.optim.SGD(param, 1e-1)
criterion = nn.BCEWithLogitsLoss()
for e in range(10000):
    out = seq_net(Variable(x))  # m*2
    loss = criterion(out, Variable(y))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e + 1, loss.data[0]))
