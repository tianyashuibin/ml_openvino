# alternative_quantization.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


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

    # 导出为ONNX
    print("导出ONNX模型...")
    dummy_input = torch.randn(1, 3, 32, 32)

    # 保存为ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        "float_model.onnx",
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        # dynamo=True
    )
    print("✓ ONNX 导出成功")

    # 使用ONNX Runtime进行训练后量化
    print("进行训练后量化...")
    try:
        quantize_dynamic(
            "float_model.onnx",
            "quantized_model.onnx",
            weight_type=QuantType.QInt8
        )
        print("训练后量化完成: quantized_model.onnx")
    except Exception as e:
        print(f"ONNX量化错误: {e}")
        print("请安装ONNX Runtime: pip install onnxruntime")
        return


if __name__ == "__main__":
    main()