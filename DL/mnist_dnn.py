import torch
from torch import optim
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
# print(use_cuda)

#设置device变量
if use_cuda:
    device = torch.device("cuda")
    print("use cuda")
else:
    device = torch.device("cpu")
    print("use cpu")

#设置对数据进行处理的逻辑
transform = transforms.Compose([
    # 将图片转化为tensor
    transforms.ToTensor(),
    # 归一化 0.1307 是均值 0.3081是标准差
    transforms.Normalize((0.1307,), (0.3081,))
])

#读取数据
datasets1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
datasets2 = datasets.MNIST('./data', train=False, download=True,transform=transform)

#设置数据加载器,设置批次大小，是否打乱数据顺序
train_loader = torch.utils.data.DataLoader(datasets1, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets2, batch_size=1000)

# for batch_idx, data in enumerate(train_loader,0):
#     input,target = data
#     #view在下一行会把我们的训练集（60000,1,28,28）转换为（60000,784）
#     x = input.view(-1,28*28)
#     #获取数据集的均值和标准差
#     x_std = x.std().item()
#     x_mean = x.mean().item()
#
# print(x_std,x_mean)

#通过自定义类构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义一个全连接层，输入为784，输出为128 -线性加权求和
        self.fc1 = nn.Linear(784, 128)
        # 定义一个dropout层，防止过拟合
        self.dropout = nn.Dropout(0.2)
        # 定义一个全连接层，输入为128，输出为10
        self.fc2 = nn.Linear(128, 10)
        # self.fc2 = nn.Linear(512, 256)

    #正向传播
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # 使用ReLU激活函数 -加入非线性变化
        x = F.relu(x)
        x =self.dropout(x)
        x = self.fc2(x)
        # 使用softmax激活函数 -输出层 把输出变成概率再取log--方便计算损失函数
        output = F.log_softmax(x, dim=1)
        return output


# 创建模型实例
model = Net().to(device)


# 定义训练模型的逻辑
def train_step(data,target,model,optimizer):
    #
    optimizer.zero_grad()
    output = model(data)
    #nll代表 negative log likely hood 负对数似然
    loss = F.nll_loss(output, target)
    #反向传播的本质就是去求梯度，然后应用梯度去调参
    loss.backward()
    optimizer.step()
    return loss

#定义测试模型的逻辑
def test_step(data,target,model,test_loss,correct):
    output = model(data)
    #累计的批次损失
    test_loss += F.nll_loss(output, target,reduction='sum').sum().item()
    #获得对数概率最大值对于的索引号，这里其实就是类别号
    pred = output.argmax(dim = 1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss,correct

#创建训练调参使用的优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
#真的分轮次训练
EPOCHES = 5

for epoch in range(EPOCHES):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        train_loss = train_step(data,target,model,optimizer)
        # 每个10个批次打印信息
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),train_loss.item()))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data,target = data.to(device),target.to(device)
            test_loss,correct = test_step(data,target,model,test_loss,correct)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy : {}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),
                                                                                  100. * correct / len(test_loader.dataset)))











