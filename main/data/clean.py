import json
import os
import re
from pathlib import Path

# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRE_DATA_DIR = Path(os.getenv("RAG_PRE_DATA_DIR", str(PROJECT_ROOT / "pre_data" / "product")))
INPUT_JSONL = os.getenv("RAG_CLEAN_INPUT", str(PRE_DATA_DIR / "all_texts.jsonl"))
OUTPUT_JSONL = os.getenv("RAG_CLEAN_OUTPUT", str(PRE_DATA_DIR / "cleaned_texts.jsonl"))

CHUNK_SIZE = 260    # 目标块长度（字符）
MIN_CHARS = 5       # 最终保留的最小字符数（太短的如“目录”建议丢弃）
# ==============================================

def normalize_text(text: str) -> str:
    """
    基础清洗
    """
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)
    # 移除不可见控制字符
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
    text = text.replace("\u3000", " ")
    return text.strip()

def split_text_recursive(text: str, max_size: int, separators=None):
    """
    段落内部的语义切分逻辑：
    如果一个段落超过 max_size，则优先按句子切分。
    优先级：句末标点 -> 句中标点 -> 空格
    """
    if separators is None:
        separators = ["。", "！", "？", "!", "?", "；", ";", "，", ",", " "]

    # 递归基：长度合适直接返回
    if len(text) <= max_size:
        return [text]
    
    # 寻找最高优先级的切分符
    selected_sep = ""
    for sep in separators:
        if sep in text:
            selected_sep = sep
            break
            
    # 如果没找到分隔符，强行切分
    if selected_sep == "":
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]
    
    # 使用分隔符切分 (保留分隔符)
    _escaped_sep = re.escape(selected_sep)
    raw_splits = re.split(f"({_escaped_sep})", text)
    
    final_chunks = []
    current_chunk = []
    current_len = 0

    for piece in raw_splits:
        if not piece:
            continue
            
        if current_len + len(piece) <= max_size:
            current_chunk.append(piece)
            current_len += len(piece)
        else:
            if current_chunk:
                doc = "".join(current_chunk).strip()
                if doc:
                    final_chunks.append(doc)
            
            current_chunk = [piece]
            current_len = len(piece)
            
            # 如果单片依然过长，递归切分
            if len(piece) > max_size:
                try:
                    sep_index = separators.index(selected_sep)
                    next_separators = separators[sep_index + 1:]
                except ValueError:
                    next_separators = []
                
                sub_doc = "".join(current_chunk) # 取出当前过长的 piece
                # 实际上 current_chunk 这里只有 piece，直接拿 piece 递归
                sub_chunks = split_text_recursive(piece, max_size, next_separators)
                
                # 将递归结果直接存入 final
                final_chunks.extend(sub_chunks)
                current_chunk = []
                current_len = 0

    # 收尾
    if current_chunk:
        doc = "".join(current_chunk).strip()
        if doc:
            final_chunks.append(doc)

    return final_chunks

def process_jsonl(input_path: str, output_path: str):
    in_path = Path(input_path)
    out_path = Path(output_path)

    # 清空输出文件
    out_path.write_text("", encoding="utf-8")
    
    processed_count = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("a", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                print(f"[WARN] 第 {line_idx} 行 JSON 解析失败: {e}")
                continue

            raw_text = record.get("text", "")
            if not raw_text or not str(raw_text).strip():
                continue

            # 1. 提取图片路径，确保为列表
            image_paths = record.get("image_paths", [])
            if image_paths is None:
                image_paths = []

            # 2. 整体规范化
            clean_full_text = normalize_text(str(raw_text))

            # 3. 步骤一：先按换行符切分出“自然段落” (Para ID 的来源)
            # 连续两个及以上换行符通常表示段落间隔，单个换行可能是排版软回车
            # 这里为了准确分段，简单的按 \n+ 切分，然后 trim
            paragraphs = re.split(r"\n+", clean_full_text)
            
            real_para_idx = 0 # 用于记录实际写入的段落ID

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # 4. 步骤二：段落内按语义切分成 Chunks
                chunks = split_text_recursive(para, CHUNK_SIZE)
                
                first_chunk_in_para = True
                chunk_counter = 0

                for chunk in chunks:
                    if len(chunk) < MIN_CHARS:
                        continue
                    
                    # =======================================================
                    # 构建输出字典，严格控制顺序
                    # =======================================================
                    out_rec = {
                        "file_path": record.get("file_path"),
                        "ext": record.get("ext"),
                        "para_id": real_para_idx,  # 当前段落号
                        "chunk_id": chunk_counter, # 当前段落内的块号
                        "text": chunk,
                        "image_paths": image_paths # 放在最后
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    
                    chunk_counter += 1
                    first_chunk_in_para = False
                
                # 只有当该段落确实产出了有效 chunk 时，para_id 才自增
                # (避免因为段落太短被过滤导致 para_id 跳号，看你需求，通常自增比较好)
                if not first_chunk_in_para:
                    real_para_idx += 1

            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"已处理 {processed_count} 条文件记录...")

    print(f"✅ 清洗完成！\n结果已保存至：{output_path}")

if __name__ == "__main__":
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
