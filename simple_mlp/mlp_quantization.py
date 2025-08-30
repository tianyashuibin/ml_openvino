import torch
import torch.nn as nn
import torch.quantization

# 1. 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleNet()
print("原始模型结构：")
print(model)

# 2. 准备量化
# 在本例中，我们使用动态量化，因为它非常简单
# 动态量化只对权重进行量化，并在推理时动态地量化激活值
# 这使得它对内存和计算都有好处，同时操作简单
model.eval()

# 3. 进行量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # 指定要量化的模块类型
    dtype=torch.qint8  # 指定量化类型为 int8
)
print("\n量化后的模型结构：")
print(quantized_model)

# 4. 将量化模型转换为 ONNX 格式
# ONNX 导出需要一个输入张量作为示例，以便追踪模型的计算图
dummy_input = torch.randn(1, 10)  # 创建一个形状为 (1, 10) 的虚拟输入

# 定义 ONNX 文件名
onnx_path = "quantized_model_int8.onnx"

try:
    torch.onnx.export(
        quantized_model,
        dummy_input,
        onnx_path,
        opset_version=14,  # 通常建议使用较新的 opset_version
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"\n模型已成功保存到 {onnx_path}")

except Exception as e:
    print(f"导出 ONNX 时出错：{e}")