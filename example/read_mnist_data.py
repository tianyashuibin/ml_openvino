import gzip
import numpy as np

# 读取标签文件
def read_labels(filename):
    with open(filename, 'rb') as f:
        # 读取 magic number 和标签数量
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        print(f"Magic Number: {magic_number}")
        print(f"Number of labels: {num_labels}")

        # 读取所有标签
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# 文件路径（根据你的实际情况修改）
file_path = 'data/MNIST/raw/train-labels-idx1-ubyte'

# 读取标签
labels = read_labels(file_path)

# 查看前20个标签
print("前20个标签:", labels[:20])



# 读取minist图像文件
def read_images(filename):
    with open(filename, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        print(f"Magic Number: {magic_number}")
        print(f"Number of images: {num_images}")
        print(f"Image shape: {rows}x{cols}")

        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images

# 图像文件路径
image_file = 'data/MNIST/raw/train-images-idx3-ubyte'
images = read_images(image_file)

# 显示第一张图像和它的标签
import matplotlib.pyplot as plt

plt.imshow(images[0], cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.show()