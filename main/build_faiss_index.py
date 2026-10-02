import json
import os
import faiss
import numpy as np
from pathlib import Path

# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
INPUT_JSONL = os.getenv("RAG_INDEX_INPUT", str(PRE_DATA_DIR / "bge-m3.jsonl"))
VECTOR_INDEX_PATH = os.getenv("RAG_INDEX_OUTPUT", str(PROJECT_ROOT / "indexes" / "product" / "faiss_index.index"))
# ==============================================

def load_embeddings_from_jsonl(input_jsonl):
    """
    从 JSONL 文件中加载 embedding 向量，并提取出需要的信息
    """
    embeddings = []
    metadata = []  # 用来存储文本对应的元数据，方便后续查找
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            emb = record.get("embedding")
            if emb is not None:
                embeddings.append(emb)
                metadata.append(record)
    return np.array(embeddings), metadata

def build_faiss_index(embeddings):
    """
    使用 FAISS 构建 **余弦相似度索引** (内积 + 归一化 = 余弦相似度)
    """
    dim = embeddings.shape[1]  # 向量的维度

    # ==================== 关键修改 1：向量归一化 ====================
    # 将所有 embedding 归一化为模长=1（L2 归一化）
    # 归一化后，内积等价于余弦相似度
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 处理可能的零向量（避免除零错误）
    embeddings[np.isnan(embeddings)] = 0.0
    # ==================== 关键修改 2：改用内积索引 ====================
    index = faiss.IndexFlatIP(dim)  # 内积索引（cosine similarity after normalization）

    # 将 embedding 添加到 FAISS 索引
    index.add(embeddings)
    return index

def save_faiss_index(index, index_path):
    """
    保存 FAISS 索引到磁盘
    """
    faiss.write_index(index, index_path)
    print(f"FAISS 余弦相似度索引已保存至: {index_path}")

def main():
    # 从 JSONL 加载 embedding 向量
    print("[INFO] 加载向量和元数据...")
    embeddings, metadata = load_embeddings_from_jsonl(INPUT_JSONL)

    # 创建 FAISS 索引（现在是余弦相似度）
    print("[INFO] 构建 FAISS 余弦相似度索引...")
    index = build_faiss_index(embeddings)

    # 保存索引
    print("[INFO] 保存 FAISS 索引...")
    save_faiss_index(index, VECTOR_INDEX_PATH)

    # 保存元数据到文件
    metadata_path = VECTOR_INDEX_PATH.replace(".index", "_metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"元数据已保存至: {metadata_path}")

if __name__ == "__main__":
    main()
