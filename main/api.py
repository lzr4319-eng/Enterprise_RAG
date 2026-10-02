from typing import Dict, Any, Generator
import json
import traceback
from pathlib import Path

# 【重要】这里不再导入 run_rag_once，而是导入 RAG 类
# 假设你在 rag_core.py 中已经封装好了 class RAG
from rag_core import RAG

# === 1. 配置与初始化区域 ===
BASE_DIR = Path(__file__).resolve().parent.parent / "indexes"

# 预加载所有知识库引擎
print("正在初始化多路知识库引擎...")
try:
    ENGINES = {
        # 默认引擎 (红色)
        "employee": RAG(
            index_path=str(BASE_DIR / "employee" / "faiss_index.index"),
            metadata_path=str(BASE_DIR / "employee" / "faiss_index_metadata.jsonl")
        ),
        # 产品信息 (蓝色)
        "product": RAG(
            index_path=str(BASE_DIR / "product" / "faiss_index.index"),
            metadata_path=str(BASE_DIR / "product" / "faiss_index_metadata.jsonl")
        ),
        # 保密信息 (紫色)
        "secret": RAG(
            index_path=str(BASE_DIR / "secret" / "faiss_index.index"),
            metadata_path=str(BASE_DIR / "secret" / "faiss_index_metadata.jsonl")
        ),
        # 专业知识 (绿色)
        "professional": RAG(
            index_path=str(BASE_DIR / "professional" / "faiss_index.index"),
            metadata_path=str(BASE_DIR / "professional" / "faiss_index_metadata.jsonl"),
            enable_fallback=True
        ),
    }
    LOAD_ERROR = None
    print("多路知识库初始化完成。")
except Exception as e:
    LOAD_ERROR = str(e)
    print(f"知识库初始化严重错误: {e}")
    ENGINES = {}

# === 2. 修改函数签名：增加 category 参数，默认值为 'employee' ===
def run_rag_session(prompt: str, category: str = "employee") -> Generator[Dict[str, Any], None, None]:
    """
    将 RAG pipeline 封装成一个“按步骤产生日志”的生成器。
    支持根据 category 切换不同的知识库。
    """
    
    # 0. 初始化检查
    if LOAD_ERROR:
        yield {
            "phase": "error",
            "gen": None,
            "message": f"系统初始化失败: {LOAD_ERROR}",
        }
        return

    # 1. 收到问题 (日志中增加显示当前使用的库)
    kb_names = {
        "employee": "员工培训库",
        "product": "产品信息库",
        "secret": "保密信息库",
        "professional": "专业知识库"
    }
    kb_display = kb_names.get(category, "未知知识库")
    
    yield {
        "phase": "meta",
        "gen": None,
        "message": f"收到问题：{prompt} (正在检索: {kb_display})",
        "best_fitness": None,
        "best_plan": None,
    }

    # 获取对应的引擎实例
    current_engine = ENGINES.get(category)
    if not current_engine:
        yield {
            "phase": "error",
            "gen": None,
            "message": f"错误：未找到类别为 '{category}' 的知识库引擎。",
        }
        return

    try:
        # 2. 执行 RAG（检索 + 生成回答）
        # 【变化点】调用实例方法 chat 或 run_once，而不是全局函数
        # 假设 RAG 类中有一个方法叫 chat 或 run_rag_once，返回结果结构与之前一致
        result = current_engine.chat(prompt)  

        refs = result.get("references", [])
        answer = result.get("answer", "")
        related_qs = result.get("related_questions", [])

        # 2.1 检索阶段：把命中的片段逐条写入日志
        if refs:
            for ref in refs:
                score_display = f"RRF:{ref.get('score', 0):.4f}"
                
                # 处理图片预览逻辑 (如果数据处理脚本正确生成了 image_paths)
                preview = ref.get('preview_text', '')
                if ref.get('type') == 'image_caption':
                    preview = "[图片内容匹配]" + preview
                    
                msg = f"检索到片段[{ref['ref_id']}] {score_display}: {ref.get('file_path')} (预览: {preview})"
                
                yield {
                    "phase": "retrieval",
                    "gen": ref["ref_id"],
                    "message": msg,
                    "best_fitness": None,
                    "best_plan": None,
                }
        else:
            yield {
                "phase": "retrieval",
                "gen": None,
                "message": "未检索到足够相关的片段，将尝试直接作答。",
                "best_fitness": None,
                "best_plan": None,
            }

        # 3. 回答阶段：输出最终回答
        yield {
            "phase": "answer",
            "gen": None,
            "message": answer,
            "best_fitness": None,
            "best_plan": None,
        }

        # 3.2 输出结构化结果
        structured = {
            "answer": answer,
            "references": refs,   
            "related_questions": related_qs 
        }
        
        yield {
            "phase": "result",
            "gen": None,
            "message": json.dumps(structured, ensure_ascii=False),
            "best_fitness": None,
            "best_plan": None,
        }

    except Exception as e:
        # 4. 出错阶段
        traceback.print_exc() 
        yield {
            "phase": "error",
            "gen": None,
            "message": f"RAG 执行出错: {str(e)}",
            "best_fitness": None,
            "best_plan": None,
        }

if __name__ == "__main__":
    # 测试代码
    print("--- 测试产品库 ---")
    for step in run_rag_session("激光打标设备增长率是多少？", category="product"):
        print(step)
        
    print("\n--- 测试员工库 ---")
    for step in run_rag_session("试用期多久？", category="employee"):
        print(step)
