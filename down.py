# download_model.py
from modelscope import snapshot_download
import os
from pathlib import Path

# 指定下载目录
PROJECT_ROOT = Path(__file__).resolve().parent
save_dir = os.getenv("RAG_MODEL_DIR", str(PROJECT_ROOT / "models"))

print("开始从魔塔下载 bge-reranker-v2-m3...")
model_dir = snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir=save_dir)

print("\n下载完成！")
print(f"请将代码中的模型路径修改为: {model_dir}")
