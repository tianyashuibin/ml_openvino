# 1. 模型改造：从内置Embedding到外部输入
import torch
import torch.nn as nn

# 改造前：内置Embedding层
class OriginalModel(nn.Module):
    def __init__(self, num_items, embed_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)  # 内置Embedding
        self.mlp = nn.Linear(embed_dim, 1)
        
    def forward(self, item_ids):
        embeddings = self.item_embedding(item_ids)
        return self.mlp(embeddings)

# 改造后：外部输入Embedding
class SplitModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Linear(embed_dim, 1)  # 仅保留预测层
        
    def forward(self, embeddings):
        return self.mlp(embeddings)  # 直接使用外部传入的Embedding


# 2. 导出并存储Embedding到Redis
import redis
import numpy as np

def export_and_store_embeddings(original_model, redis_client, prefix="item_"):
    # 从原始模型中导出Embedding参数
    embeddings = original_model.item_embedding.weight.detach().numpy()
    num_items, embed_dim = embeddings.shape
    
    # 批量存储到Redis
    for i in range(num_items):
        item_id = f"{prefix}{i}"
        # 将numpy向量序列化为字节流（节省空间）
        embedding_bytes = embeddings[i].astype(np.float32).tobytes()
        redis_client.set(item_id, embedding_bytes)
    
    print(f"已存储 {num_items} 个Embedding，维度: {embed_dim}")


# 3. 在线推理：查询Embedding并预测
def predict_with_split_embedding(item_id, model, redis_client, embed_dim=64, prefix="item_"):
    # 从Redis查询Embedding
    embedding_bytes = redis_client.get(f"{prefix}{item_id}")
    
    if not embedding_bytes:
        # 处理冷启动：返回默认向量（如全零向量）
        embedding = np.zeros(embed_dim, dtype=np.float32)
    else:
        # 反序列化字节流为numpy向量
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    
    # 转换为模型输入格式
    embedding_tensor = torch.tensor(embedding).unsqueeze(0)  # 增加批次维度
    
    # 模型预测（关闭梯度计算）
    with torch.no_grad():
        model.eval()
        score = model(embedding_tensor)
    
    return score.item()


# 示例运行流程
if __name__ == "__main__":
    # 初始化参数
    num_items = 10000  # 物品数量
    embed_dim = 64     # Embedding维度
    
    # 1. 初始化原始模型并训练（此处省略训练过程）
    original_model = OriginalModel(num_items, embed_dim)
    
    # 2. 连接Redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # 3. 导出并存储Embedding
    export_and_store_embeddings(original_model, redis_client)
    
    # 4. 加载拆分后的模型（仅包含MLP参数）
    split_model = SplitModel(embed_dim)
    
    # 5. 在线预测示例
    item_id = 123  # 待预测的物品ID
    prediction = predict_with_split_embedding(item_id, split_model, redis_client)
    print(f"物品 {item_id} 的预测得分: {prediction:.4f}")
