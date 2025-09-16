import argparse
import torch
import sys


def check_model_types(model_path):
    # 加载 TorchScript 模型（关键修改：用 torch.jit.load 替代 torch.load）
    try:
        # TorchScript 模型加载需指定 map_location，避免设备不匹配
        model = torch.jit.load(model_path, map_location='cpu')
        print("✅ 成功加载 TorchScript 模型")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {model_path}")
        return
    except Exception as e:
        print(f"❌ 加载 TorchScript 模型时出错：{str(e)}")
        return

    # 提取模型的 state_dict（权重字典）
    # 注意：TorchScript 模型的 state_dict 键名可能带 "module." 前缀（若训练时用了 DataParallel）
    state_dict = model.state_dict()

    if not state_dict:
        print("❌ 未从模型中提取到 state_dict（可能是纯推理图无可训练参数）")
        return

    # 遍历参数，打印关键信息
    print(f"\n📊 模型共包含 {len(state_dict)} 个可训练参数/缓冲区：")
    print("-" * 80)
    for idx, (name, param) in enumerate(state_dict.items(), 1):
        print(f"参数 {idx}:")
        print(f"  名称: {name}")
        print(f"  数据类型: {param.dtype}")  # 如 torch.float32、torch.int64
        print(f"  形状: {param.shape}")      # 如 (128, 64)（权重）、(128,)（偏置）
        print(f"  设备: {param.device}")      # 验证是否加载到 CPU
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='查看 PyTorch/TorchScript 模型的参数信息')
    parser.add_argument('model_path', type=str, help='TorchScript 模型文件路径（.pt/.pth）')
    args = parser.parse_args()

    check_model_types(args.model_path)


if __name__ == "__main__":
    main()
