import json
import os
import re
from pathlib import Path

# ================= 配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
INPUT_FILE = Path(os.getenv("RAG_FIT_INPUT", str(PRE_DATA_DIR / "cleaned_texts.jsonl")))
OUTPUT_FILE = Path(os.getenv("RAG_FIT_OUTPUT", str(PRE_DATA_DIR / "cleaned_texts_fit.jsonl")))
# ===========================================

def clean_text_content(text):
    """
    清洗text字段的具体逻辑
    """
    if not text:
        return ""

    # 0. [新增] 清理 HTML 标签
    # 匹配 <...> 形式的内容，例如 <div>, <imgSrc...>, <br/> 等
    # 替换为空格以防文字粘连，后续步骤会合并多余空格
    text = re.sub(r'<[^>]+>', ' ', text)

    # 1. 清理 PPT 等转换产生的 Slide 分页标记
    # 匹配模式：连字符 + Slide + 数字 + 连字符
    # 例如: "--- Slide 12 ---", "-- Slide 1 --", "--- slide 8 ---"
    text = re.sub(r'-+\s*Slide\s*\d+\s*-+', ' ', text, flags=re.IGNORECASE)

    # 2. 清理表格评分/占位符
    # 匹配连续的 空白/Tab/★/☆ 组合，例如: \t★\t☆\t☆\t★
    text = re.sub(r'[\t\s]*[★☆]+[\t\s★☆]*', ' ', text)

    # 3. 清理特殊目录头
    # 匹配 罗马数字 + ● 符号，例如: Ⅰ●, Ⅱ●
    text = re.sub(r'[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+●', ' ', text)

    # 4. 清理多余的空白字符
    # 将连续的 \t, \n, 空格 合并为一个空格
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def process_single_file(input_path, output_path):
    # 确保输出文件的父目录存在
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    print(f"正在读取文件: {input_path}")
    
    valid_count = 0   # 记录保留的行数
    dropped_count = 0 # 记录被删除的行数
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                stripped_line = line.strip()
                # 跳过文件本身的空行
                if not stripped_line:
                    continue
                
                try:
                    # 解析JSON
                    item = json.loads(stripped_line)
                    
                    # 获取原始文本，确保它是字符串
                    raw_text = item.get("text", "")
                    if not isinstance(raw_text, str):
                        raw_text = str(raw_text)

                    # === 核心逻辑修改开始 ===
                    
                    # 执行清洗
                    cleaned_text = clean_text_content(raw_text)
                    
                    # 判断：如果清洗后内容为空，直接跳过（相当于删除这条数据）
                    if not cleaned_text:
                        dropped_count += 1
                        continue 
                    
                    # 如果不为空，更新字典中的 text 字段
                    item["text"] = cleaned_text
                    
                    # === 核心逻辑修改结束 ===

                    # 写入新文件 (ensure_ascii=False 保证中文不被转义)
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    valid_count += 1
                    
                except json.JSONDecodeError:
                    print(f"警告: 发现无法解析的JSON行，已跳过。")
                    continue

        print(f"-" * 30)
        print(f"处理完成！")
        print(f"✅ 保留有效数据: {valid_count} 行")
        print(f"🗑️ 过滤无效数据: {dropped_count} 行 (text为空、slide标记或html代码)")
        print(f"输出文件已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    process_single_file(INPUT_FILE, OUTPUT_FILE)
