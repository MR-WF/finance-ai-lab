"""
从 ModelScope（魔搭，国内稳定源）下载本项目所需的两个模型到本地 ./models 目录，
彻底绕开 HuggingFace / hf-mirror（国内常超时）。
下载后 rag_w3.py 会自动兜底：在线加载连续失败 5 次后切换 ./models 本地离线加载；
若想跳过在线尝试（明知网络不可用），可在 rag_w3.py 把 FORCE_LOCAL=True。

用法：
  pip install modelscope
  python download_models_modelscope.py
"""
import os
from modelscope import snapshot_download

os.makedirs("models", exist_ok=True)

MODELS = {
    # embedding 模型（W2/W3 共用，中文句向量）
    "BAAI/bge-small-zh-v1.5": "models/bge-small-zh-v1.5",
    # 重排模型（W3 引入，对候选片段精排）
    "BAAI/bge-reranker-v2-m3": "models/bge-reranker-v2-m3",
}

for repo_id, local_dir in MODELS.items():
    print(f"[下载] {repo_id}  ->  {local_dir}")
    snapshot_download(repo_id, local_dir=local_dir)
    print(f"[完成] {repo_id}\n")

print("==== 全部模型已下载到 ./models，可离线运行 rag_w3.py ====")
