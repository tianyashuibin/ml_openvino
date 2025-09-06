import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torchvision import datasets, transforms

# 确保 PyTorch 支持 OpenMP，这是量化所需要的
torch.backends.quantized.engine = 'qnnpack'

# 定义一个简单的 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 为了更好地进行量化，可以把每个模块定义为独立的子模块
        # 这样在fuse_modules时路径更清晰
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# 准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 训练模型（这里只进行简单的训练以获取一个可用的模型）
def train_model(model, train_loader, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("模型训练完成.")

# 实例化模型并训练
model_fp32 = LeNet5()
train_model(model_fp32, train_loader)

# --- 训练后 INT8 量化 ---
# 1. 准备模型进行量化，注意这里使用prepare_for_inference
# 2. 设置量化配置：使用 'qnnpack' 引擎的默认配置
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

# 3. 准备模型：这将插入观察器到支持量化的层
torch.quantization.prepare_for_inference(model_fp32, inplace=True)

# 4. 校准模型
print("开始校准模型...")
model_fp32.eval()
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if i >= 10:
            break
        model_fp32(data)
print("模型校准完成.")

# 5. 转换模型
model_int8 = torch.quantization.convert(model_fp32, inplace=True)
print("模型已转换为 INT8 量化版本.")

# --- 保存为 ONNX 格式 ---
dummy_input = torch.randn(1, 1, 28, 28)
onnx_file_path = "quantized_lenet.onnx"
torch.onnx.export(model_int8,
                  dummy_input,
                  onnx_file_path,
                  opset_version=13,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})
print(f"INT8 量化模型已成功保存为 {onnx_file_path}")