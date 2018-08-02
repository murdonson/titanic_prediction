import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from PIL import Image
from torch import nn
from torch.autograd import Variable

# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)
a_data,a_label=train_set[0]

a_data=np.array(a_data,dtype='float32')

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)



from torch.utils.data import DataLoader
# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

a, a_label=next(iter(train_data))

net=nn.Sequential(
    nn.Linear(784,400),
    nn.ReLU(),
    nn.Linear(400,200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),1e-1)

losses=[]
acces=[]
eval_losses=[]
eval_acces=[]

for e in range(20):
    train_loss=0
    train_acc=0
    net.train()
    for im, label in train_data:
            im = Variable(im)
            label = Variable(label)
            # 前向传播
            out = net(im)
            loss = criterion(out, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.data[0]
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().data[0]
            acc = num_correct / im.shape[0]
            train_acc += acc


