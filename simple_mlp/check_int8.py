# check_onnx_dtype.py
import onnx

# 加载量化后的 ONNX 模型
#model = onnx.load("quantized_model_int8.onnx")
model = onnx.load("quantized_static_model_int8.onnx")

# 遍历所有 initializer（即权重）
for init in model.graph.initializer:
    print(f"权重: {init.name} -> 数据类型: {init.data_type} (shape: {list(init.dims)})")
