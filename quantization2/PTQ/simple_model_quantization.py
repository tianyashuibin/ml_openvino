import torch
import torch.nn as nn
import torch.quantization

# 1. 定义简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. 创建模型
model_fp32 = SimpleModel()

# 3. 配置静态量化
model_fp32.qconfig = torch.quantization.get_default_qconfig("qnnpack")

# 4. 插入量化伪节点
model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

# 5. 校准（这里用几组随机数据代替实际数据集）
for _ in range(100):
    dummy_input = torch.randn(1, 10)
    model_prepared(dummy_input)

# 6. 转换为量化模型
model_int8 = torch.quantization.convert(model_prepared, inplace=False)

# 7. 导出 ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model_int8,
    dummy_input,
    "simple_model_int8_static.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)

print("✅ 已保存为 simple_model_int8_static.onnx")
