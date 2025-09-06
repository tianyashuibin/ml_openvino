import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
import torch.ao.quantization.quantize_fx as quantize_fx
# from openvino.tools import mo
# from openvino.runtime import serialize


# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    # 设置设备
    device = torch.device('cpu')

    # 创建训练数据
    x_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 10, (100,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

    # 训练模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("开始训练模型...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/3, Loss: {total_loss / len(train_loader):.4f}')

    # 使用FX图模式量化
    print("开始量化模型...")

    # 设置量化配置
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")  # 使用qnnpack后端（适用于CPU）

    # 准备示例输入
    example_inputs = (torch.randn(1, 3, 32, 32),)

    # 准备模型进行量化
    model.eval()
    prepared_model = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs
    )

    # 校准模型（使用训练数据进行校准）
    print("校准模型...")
    with torch.no_grad():
        for inputs, _ in train_loader:
            prepared_model(inputs.to(device))

    # 转换为量化模型
    quantized_model = convert_fx(prepared_model)

    print("模型量化完成")

    # 导出ONNX
    print("导出ONNX模型...")
    dummy_input = torch.randn(1, 3, 32, 32)

    # 设置模型为评估模式
    quantized_model.eval()

    # 导出为ONNX格式
    torch.onnx.export(
        quantized_model,
        dummy_input,
        "quantized_model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("ONNX模型已保存: quantized_model.onnx")

    # 转换为OpenVINO格式
    # try:
    #     print("转换为OpenVINO格式...")
    #     ov_model = mo.convert_model("quantized_model.onnx")
    #     serialize(ov_model, "quantized_model.xml")
    #     print("OpenVINO模型已保存: quantized_model.xml")
    # except Exception as e:
    #     print(f"OpenVINO转换错误: {e}")
    #     print("请确保已安装OpenVINO开发工具: pip install openvino-dev")
    #
    # # 测试量化模型
    # print("测试量化模型...")
    # with torch.no_grad():
    #     test_input = torch.randn(1, 3, 32, 32)
    #     output = quantized_model(test_input)
    #     print(f"量化模型输出形状: {output.shape}")
    #     print(f"预测类别: {torch.argmax(output, dim=1).item()}")


if __name__ == "__main__":
    main()