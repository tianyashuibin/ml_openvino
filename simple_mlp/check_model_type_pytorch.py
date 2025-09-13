import argparse
import torch
import sys


def check_model_types(model_path):
    # 加载指定的ckpt文件
    try:
        ckpt = torch.load(model_path, map_location='cpu')  # 强制在CPU上加载
    except FileNotFoundError:
        print(f"错误：找不到文件 {model_path}")
        return
    except Exception as e:
        print(f"加载文件时出错：{str(e)}")
        return

    # 通常模型权重存储在'state_dict'键下，如果是直接保存的模型可以直接使用ckpt
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # 遍历状态字典，查看每个层的参数名称和数据类型
    for name, param in state_dict.items():
        print(f"参数名称: {name}")
        print(f"数据类型: {param.dtype}")
        print(f"形状: {param.shape}")
        print("---")


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='查看PyTorch模型检查点文件中各层的数据类型')
    parser.add_argument('model_path', type=str, help='模型检查点文件的路径和名称')
    args = parser.parse_args()

    check_model_types(args.model_path)


if __name__ == "__main__":
    main()
