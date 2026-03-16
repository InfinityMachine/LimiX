import os, sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from inference.predictor import LimiXPredictor


def clean_data(df):
    # 删除所有列中缺少数据的行
    df = df.dropna()
    return df


# 已从 URI 中加载变量“df”: \content\data.csv
df = pd.read_csv(r"data.csv", engine="pyarrow")

df_clean = clean_data(df.copy())
df_clean.head()

# 1) 读数据
df = df_clean

# 2) 选列：y 是 PM2.5，其余是 X
cat_cols = ["PROVINCE", "CITY", "COUNTY"]
num_cols = ["AET", "ppt", "tem", "wind", "NOX", "SO2", "fertilzier", "manure"]
y_col = "PM2.5"

# 3) 编码分类列 -> 整数id（更稳）
for c in cat_cols:
    df[c] = df[c].astype("category").cat.codes

# 4) 组装 X, y（按列顺序：先 cat 后 num）
X = df[cat_cols + num_cols].to_numpy(dtype=np.float32)
y = df[y_col].to_numpy(dtype=np.float32)

# 5) 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) 标准化 y（按 README 回归示例）
y_mean = y_train.mean()
y_std = y_train.std() + 1e-8
y_train_norm = (y_train - y_mean) / y_std

# 7) 下载模型权重（repo_id 以你能下载到的为准）
model_path = "modelCache/LimiX-2M.ckpt"

# 8) 选推理配置（用你本地实际存在的那个）
# cfg_path = f"{ROOT_DIR}/config/reg_default_2M_taar.json"  # 或 reg_default_noretrieval.json
# cfg_path = f"{ROOT_DIR}/config/reg_default_noretrieval.json"
cfg_path = f"config/reg_default_2M_retrieval.json"

# 9) 创建回归 predictor（支持传 categorical_features_indices）
model = LimiXPredictor(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_path=model_path,
    inference_config=cfg_path,
    categorical_features_indices=[0, 1, 2],  # PROVINCE/CITY/COUNTY 在 X 的前三列
)


# 10) 预测
y_pred_norm = model.predict(X_train, y_train_norm, X_test, task_type="Regression")

# 有的实现会返回 torch.Tensor，这里统一转 numpy
if hasattr(y_pred_norm, "detach"):
    y_pred_norm = y_pred_norm.detach().cpu().numpy()

# 11) 反标准化回真实 PM2.5
y_pred = y_pred_norm * y_std + y_mean

from sklearn.metrics import root_mean_squared_error, r2_score

rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)
print("R2:", r2_score(y_test, y_pred))
