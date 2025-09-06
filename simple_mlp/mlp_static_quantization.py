import torch
import torch.nn as nn
import torch.onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod, QuantFormat
from onnxruntime.quantization.calibrate import create_calibrator, CalibrationDataReader
import onnx
import numpy as np

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


# --- 新增的静态量化步骤 ---
# 3. 创建一个 CalibrationDataReader 类
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        # 创建一个包含代表性数据的迭代器
        self.data_for_calibration = []
        for _ in range(10): # 使用10个随机样本作为校准数据
            self.data_for_calibration.append({'input': np.random.randn(1, 10).astype(np.float32)})
        self.data_reader = iter(self.data_for_calibration)
        
    def get_next(self):
        # 必须实现这个方法，它返回下一批数据
        try:
            return next(self.data_reader)
        except StopIteration:
            return None

    def rewind(self):
        # 必须实现这个方法，用于重置数据读取器
        self.data_reader = iter(self.data_for_calibration)

# 4. 使用 ONNX Runtime 对导出的 ONNX 模型进行静态量化
quantized_onnx_path = "quantized_static_model_int8.onnx"
try:
    # 创建一个 MyCalibrationDataReader 实例
    calibration_data_reader = MyCalibrationDataReader()
    
    # 执行静态量化
    quantize_static(
        onnx_float_path,
        quantized_onnx_path,
        calibration_data_reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"模型已成功进行静态量化并保存到 {quantized_onnx_path}")
except Exception as e:
    print(f"使用 ONNX Runtime 进行静态量化时出错：{e}")
