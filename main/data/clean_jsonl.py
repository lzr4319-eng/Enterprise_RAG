

import json
import re
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
input_file = os.getenv("RAG_IMAGE_TAG_CLEAN_INPUT", str(PRE_DATA_DIR / "cleaned_texts.jsonl"))
temp_file = input_file + ".tmp"

# 1. 定义图片标签的正则
pattern = r"<div style=\"text-align: center;\"><img src=\"[^\"]*\" alt=\"Image\" width=\"[^\"]*%\" \/><\/div>"

with open(input_file, "r", encoding="utf-8") as fin, open(temp_file, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            data = json.loads(line)
            if "text" in data and isinstance(data["text"], str):
                original_text = data["text"]
                
                # 核心逻辑：将图片标签替换为 " 换行符 "
                # 加一个空格是为了防止原本紧挨着的文字在特定渲染下粘连
                text_replaced = re.sub(pattern, "\n", original_text)
                
                # 优化逻辑：将连续的多个换行符（比如\n\n\n）合并为单个换行符，保持整洁
                # 同时也去除行首行尾的多余空白
                text_cleaned = re.sub(r"\n+", "\n", text_replaced).strip()
                
                data["text"] = text_cleaned
                
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print(f"警告：跳过无效JSON行: {line[:50]}...")

os.replace(temp_file, input_file)
print(f"处理完成！已将图片标签替换为换行符，并整理了格式。文件：{input_file}")
