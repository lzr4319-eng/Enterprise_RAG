import json
import os
from pathlib import Path
import time

import ollama  # pip install ollama

# ==================== 配置区 ====================
MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "bge-m3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
INPUT_JSONL = os.getenv("RAG_EMBED_INPUT", str(PRE_DATA_DIR / "cleaned_texts.jsonl"))
OUTPUT_JSONL = os.getenv("RAG_EMBED_OUTPUT", str(PRE_DATA_DIR / "bge-m3.jsonl"))
ERROR_LOG = os.getenv("RAG_EMBED_ERROR_LOG", str(PRE_DATA_DIR / "bge-m3.log"))

PRINT_EVERY = 100  # 每处理多少条打印一次进度
# ==============================================

def embed_text(text: str):
    """
    调用 Ollama 的 embedding 接口，返回向量（list[float]）
    """
    resp = ollama.embeddings(model=MODEL_NAME, prompt=text)
    return resp["embedding"]

def main():
    in_path = Path(INPUT_JSONL)
    out_path = Path(OUTPUT_JSONL)
    err_path = Path(ERROR_LOG)

    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    # 清空输出文件和错误日志
    out_path.write_text("", encoding="utf-8")
    err_path.write_text("", encoding="utf-8")

    total = 0
    t0 = time.time()

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("a", encoding="utf-8") as fout, \
         err_path.open("a", encoding="utf-8") as ferr:

        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                ferr.write(f"[JSON ERROR] line {line_idx}: {e}\n")
                continue

            text = str(record.get("text", "")).strip()
            if not text:
                continue

            try:
                emb = embed_text(text)
            except Exception as e:
                ferr.write(f"[EMBED ERROR] line {line_idx}: {e}\n")
                ferr.flush()
                continue

            out_rec = {
                # 保留原来的所有字段
                **record,
                # 新增 embedding 字段
                "embedding": emb,
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            total += 1
            if total % PRINT_EVERY == 0:
                dt = time.time() - t0
                print(f"[INFO] embedded {total} records, elapsed {dt:.1f}s")

    dt = time.time() - t0
    print(f"✅ 完成向量化: {total} 条，耗时 {dt:.1f}s")
    print(f"➡ 输出文件: {OUTPUT_JSONL}")
    print(f"⚠ 错误日志: {ERROR_LOG}")

if __name__ == "__main__":
    main()
