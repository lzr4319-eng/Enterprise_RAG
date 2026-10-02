import json
import os
from pathlib import Path

# ================= 配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
INPUT_FILE = Path(os.getenv("RAG_IMAGE_CLEAN_INPUT", str(PRE_DATA_DIR / "cleaned_texts_fit.jsonl")))
# 输出文件路径 (为了安全，先输出到一个新文件)
OUTPUT_FILE = Path(os.getenv("RAG_IMAGE_CLEAN_OUTPUT", str(INPUT_FILE.with_name(INPUT_FILE.stem + "_no_imgs" + INPUT_FILE.suffix))))
# ===========================================

def clear_image_paths_field(input_path, output_path):
    print(f"正在读取文件: {input_path}")
    print(f"正在处理 image_paths 字段...")
    
    count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                try:
                    # 1. 解析 JSON
                    item = json.loads(stripped_line)
                    
                    # 2. 强制将 image_paths 字段设为空字符串
                    # 如果原字段是列表 []，现在也会变成字符串 ""
                    # 如果你需要变成空列表 []，请将下面引号里的内容改为 []
                    item["image_paths"] = []
                    
                    # 3. 写入新文件
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    
                except json.JSONDecodeError:
                    print(f"跳过无法解析的行")
                    continue

        print(f"-" * 30)
        print(f"处理完成！")
        print(f"共处理数据: {count} 行")
        print(f"结果已保存至: {output_path}")
        print(f"\n💡 如果确认无误，可以使用以下命令覆盖原文件：")
        print(f"mv {output_path} {input_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    clear_image_paths_field(INPUT_FILE, OUTPUT_FILE)
