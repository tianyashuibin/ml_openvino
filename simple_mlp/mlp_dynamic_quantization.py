import torch
import torch.nn as nn
import torch.quantization
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. 定义并实例化一个浮点数模型
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

model = SimpleNet()
model.eval()

# 2. 将浮点数模型导出为 ONNX
onnx_float_path = "model_float32.onnx"
dummy_input = torch.randn(1, 10)

try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_float_path,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"原始（浮点数）模型已成功保存到 {onnx_float_path}")
except Exception as e:
    print(f"导出 ONNX 时出错：{e}")

# 3. 使用 ONNX Runtime 对导出的 ONNX 模型进行动态量化
quantized_onnx_path = "quantized_model_int8.onnx"
try:
    # 使用 onnxruntime.quantization.quantize_dynamic
    # 它可以对ONNX模型中的权重和激活进行int8动态量化
    quantize_dynamic(
        onnx_float_path,
        quantized_onnx_path,
        weight_type=QuantType.QInt8
    )
    print(f"模型已成功进行动态量化并保存到 {quantized_onnx_path}")
except Exception as e:
    print(f"使用 ONNX Runtime 进行量化时出错：{e}")
