# mnist_mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),                    # 转为 Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化：MNIST 均值和标准差
])

# 下载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义神经网络模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # 输入 -> 隐藏层
        self.fc2 = nn.Linear(128, 10)    # 隐藏层 -> 输出
        self.relu = nn.ReLU()            # 激活函数
        self.dropout = nn.Dropout(0.2)   # 防止过拟合

    def forward(self, x):
        x = x.view(-1, 784)              # 展平图像 (N, 1, 28, 28) -> (N, 784)
        x = self.relu(self.fc1(x))       # 线性 + 激活
        x = self.dropout(x)
        x = self.fc2(x)                  # 输出 logits
        return x

model = MLP()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()        # 适用于多分类
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()            # 梯度清零
        output = model(data)             # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward()                  # 反向传播
        optimizer.step()                 # 更新参数

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                  f'Loss: {loss.item():.6f}')

# 5. 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():                # 不计算梯度
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 6. 训练主循环
if __name__ == '__main__':
    for epoch in range(1, 6):  # 训练 5 个 epoch
        train(epoch)
        test()

    # 可视化一个测试样本
    data, target = test_dataset[0]
    output = model(data.unsqueeze(0))
    pred = output.argmax().item()

    plt.imshow(data.numpy().squeeze(), cmap='gray')
    plt.title(f'Label: {target}, Predicted: {pred}')
    plt.show()