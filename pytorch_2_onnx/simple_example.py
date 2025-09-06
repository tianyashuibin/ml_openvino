import torch
import torch.nn as nn
import os

# 1. 定义一个简单的 PyTorch 模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 14 * 14, 10)  # 假设输入是 28x28

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# 2. 实例化模型并加载权重 (如果需要)
model = SimpleModel()
# model.load_state_dict(torch.load('your_model.pth')) # 如果有预训练权重

# 3. 设置模型为评估模式
model.eval()

# 4. 创建一个示例输入张量
# 形状必须和模型实际的输入形状一致，例如这里是 (batch_size, channels, height, width)
dummy_input = torch.randn(1, 1, 28, 28)

# 5. 定义 ONNX 文件的保存路径
onnx_path = "simple_model.onnx"

# 6. 调用 torch.onnx.export() 进行转换
print(f"正在将模型转换为 ONNX 格式并保存到 {onnx_path}...")
torch.onnx.export(
    model,                      # 待转换的 PyTorch 模型
    dummy_input,                # 示例输入张量
    onnx_path,                  # ONNX 模型的保存路径
    export_params=True,         # 是否导出模型的参数
    opset_version=12,           # ONNX 的操作集版本，建议使用较新版本
    do_constant_folding=True,   # 是否执行常量折叠优化
    input_names=['input'],      # 输入张量的名称，用于可读性
    output_names=['output'],    # 输出张量的名称
    dynamic_axes={              # 如果需要支持动态 batch size 或其他维度
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("模型转换成功！")

# 7. (可选) 验证 ONNX 模型
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型已成功验证！")
except ImportError:
    print("未安装 onnx 库，跳过模型验证。可以使用 pip install onnx 安装。")
except onnx.checker.ValidationError as e:
    print(f"ONNX 模型验证失败: {e}")

