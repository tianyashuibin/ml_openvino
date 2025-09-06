import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
# from torchao.quantization import quantize_
from torchvision import datasets, transforms

# 确保 PyTorch 支持 OpenMP，这是量化所需要的
torch.backends.quantized.engine = 'qnnpack'

# 定义一个简单的 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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
# 1. 准备模型：合并 Conv-BN-ReLU 模块，这能提升量化后的性能
model_fp32.fuse_model = torch.quantization.fuse_modules(model_fp32, [['features.0', 'features.1'], ['features.3', 'features.4']])

# 2. 设置量化配置：使用 `qconfig_dynamic` 进行动态量化
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model_fp32, inplace=True)

# 3. 校准模型：使用校准数据集（这里用测试集的前几批次）
print("开始校准模型...")
model_fp32.eval()
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if i >= 10:  # 只校准10个批次
            break
        model_fp32(data)
print("模型校准完成.")

# 4. 转换模型：将模型从浮点数转换为量化版本
model_int8 = torch.quantization.convert(model_fp32, inplace=True)
print("模型已转换为 INT8 量化版本.")
# 保存模型
torch.save(model_int8.state_dict(), "quantized_lenet.pt")
print("模型已保存为 quantized_lenet.pt")

# --- 保存为 ONNX 格式 ---
# 保存量化后的模型为 ONNX，需要指定输入形状
# dummy_input = torch.randn(1, 1, 28, 28)
# onnx_file_path = "quantized_lenet.onnx"
# torch.onnx.export(model_int8,
#                   dummy_input,
#                   onnx_file_path,
#                   opset_version=13,
#                   input_names=['input'],
#                   output_names=['output'],
#                   dynamic_axes={'input': {0: 'batch_size'}})
# print(f"INT8 量化模型已成功保存为 {onnx_file_path}")