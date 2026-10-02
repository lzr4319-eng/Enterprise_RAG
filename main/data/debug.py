import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
INPUT_JSONL = os.getenv("RAG_DEBUG_INPUT", str(PRE_DATA_DIR / "cleaned_texts.jsonl"))
OUTPUT_JSONL = os.getenv("RAG_DEBUG_OUTPUT", str(PRE_DATA_DIR / "with_images.jsonl"))

records_with_images = []

# 逐行读取 JSONL
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"ERROR parsing line: {e}")
            continue

        # 过滤：image_paths 存在且非空列表
        if data.get("image_paths") and len(data["image_paths"]) > 0:
            records_with_images.append(data)

# 保存提取结果
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records_with_images:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ 提取完成：共 {len(records_with_images)} 条记录")
print(f"📄 保存到 {OUTPUT_JSONL}")
