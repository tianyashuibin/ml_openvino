# complete_quantization_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
# from openvino.tools import mo
# from openvino.runtime import serialize


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

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
        x = self.quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x


def main():
    # 创建训练数据
    x_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 10, (100,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

    # 训练模型
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(3):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 量化模型
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # 校准
    model.eval()
    with torch.no_grad():
        for inputs, _ in train_loader:
            model(inputs)

    quantized_model = torch.quantization.convert(model, inplace=False)

    # 导出ONNX
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        quantized_model,
        dummy_input,
        "quantized_model.onnx",
        opset_version=13
    )

    # 转换为OpenVINO格式
    # ov_model = mo.convert_model("quantized_model.onnx")
    # serialize(ov_model, "quantized_model.xml")
    #
    # print("量化管道完成！")


if __name__ == "__main__":
    main()