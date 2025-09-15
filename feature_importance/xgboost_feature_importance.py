import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
# 在非交互式环境中使用Agg后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 1. 生成模拟数据（模拟推荐系统特征）
# 假设50个特征，其中20个是有效特征
X, y = make_classification(
    n_samples=10000,  # 样本数
    n_features=50,    # 特征数
    n_informative=20, # 有效特征数
    random_state=42
)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 转换为XGBoost的DMatrix格式（高效存储）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 3. 设置XGBoost参数（以二分类任务为例）
params = {
    'objective': 'binary:logistic',  # 二分类目标函数
    'eval_metric': 'auc',            # 评估指标（AUC）
    'max_depth': 3,                  # 树深度
    'learning_rate': 0.1,            # 学习率
    'random_state': 42
}

# 4. 训练XGBoost模型
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,  # 迭代次数
    evals=[(dtest, 'eval')],  # 验证集
    early_stopping_rounds=10,  # 早停机制（防止过拟合）
    verbose_eval=10  # 每10轮打印一次评估结果
)

# 5. 获取特征重要性（三种方式）
# 方式1：weight（默认，特征被用于分裂的总次数）
importance_weight = model.get_score(importance_type='weight')
# 方式2：gain（特征带来的总增益）
importance_gain = model.get_score(importance_type='gain')
# 方式3：cover（特征涉及的样本权重总和）
importance_cover = model.get_score(importance_type='cover')

# 6. 格式化并排序特征重要性（以gain为例）
# 将字典转换为DataFrame并排序
importance_df = pd.DataFrame(
    list(importance_gain.items()),
    columns=['feature', 'gain_importance']
).sort_values(by='gain_importance', ascending=False)

print("按增益排序的特征重要性：")
print(importance_df.head(10))  # 打印前10个最重要的特征

# 7. 可视化特征重要性（Top 15）
print("正在生成特征重要性可视化图表...")
xgb.plot_importance(
    model,
    importance_type='gain',  # 选择增益作为衡量标准
    max_num_features=15,     # 只显示前15个特征
    title='Feature Importance (Gain)',
    xlabel='Total Gain'
)

# 创建保存图表的目录
output_dir = 'feature_importance_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存图表到文件
plt.savefig(f'{output_dir}/feature_importance_gain.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"特征重要性图表已保存到：{output_dir}/feature_importance_gain.png")

# 8. 根据重要性筛选特征（例如：保留前20个特征）
top_k = 20
selected_features = importance_df['feature'].head(top_k).tolist()
# 注意：XGBoost的特征名称默认是f0, f1, ..., fn，需转换为索引
selected_indices = [int(f[1:]) for f in selected_features]  # 提取数字索引

# 9. 输出筛选后的特征索引
print(f"\n选中的前{top_k}个特征索引：", selected_indices)

# 10. 将特征重要性数据保存到CSV文件
importance_df.to_csv(f'{output_dir}/feature_importance_data.csv', index=False)
print(f"特征重要性数据已保存到：{output_dir}/feature_importance_data.csv")

# 11. 额外的模型评估信息
print("\n模型训练完成！")
print(f"最佳迭代轮数: {model.best_iteration}")
print(f"最佳验证分数: {model.best_score}")
