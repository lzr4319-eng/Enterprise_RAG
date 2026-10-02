import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import faiss
import numpy as np
import requests
import jieba
from rank_bm25 import BM25Okapi

try:
    import ollama
except ImportError:
    ollama = None

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None

# ========= 全局配置 =========

API_GENERATION_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").rstrip("/")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
EMB_MODEL = "bge-m3" 

# 【检索配置】
TOP_K = 50           
BM25_TOP_K = 50      
RRF_K = 60           
RRF_CANDIDATES = 20  
FINAL_TOP_K = 10      

# 本地模型绝对路径
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_PATH", "")

# 【核心修改区：双重过滤机制】
# 1. 绝对阈值：分数必须大于 -2.0 (过滤完全不相关的)
RERANK_THRESHOLD = -2.0 

# 2. 【新增】相对分差阈值：如果某个片段的分数比第一名低 3.5 分，直接丢弃
# 比如第一名 7.2 分，那么低于 3.7 分的全部扔掉（这样就能干掉 3.1 分的 0780 型号）
RERANK_GAP_THRESHOLD = 3.5

CONTEXT_MAX_CHARS = 12000 
MAX_IMGS_PER_CHUNK = 1

DEFAULT_IMAGE_DIRS = [
    str(Path(__file__).resolve().parent / "static" / "img"),
]

class RAG:
    def __init__(self, index_path: str, metadata_path: str, image_source_dirs: List[str] = None,
                 enable_fallback: bool = False):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.image_source_dirs = image_source_dirs if image_source_dirs else DEFAULT_IMAGE_DIRS
        
        self.index = None
        self.metadata = []
        self.bm25 = None
        self.demo_bm25_only = False
        self.load_error = None
        self.enable_fallback = enable_fallback
        
        self.reranker = None
        if FlagReranker and RERANK_MODEL_NAME and Path(RERANK_MODEL_NAME).exists():
            print(f"正在加载 Rerank 模型: {RERANK_MODEL_NAME} ...")
            try:
                self.reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)
                print("Reranker 模型加载成功！(Mode: FP16)")
            except Exception as e:
                print(f"Reranker 加载失败，将使用 RRF 排序: {e}")
        else:
            print("未配置本地 Reranker，使用 RRF 排序。")

        self._load_resources()

    def _load_resources(self):
        print(f"正在加载知识库: {os.path.basename(self.index_path)} ...")
        try:
            if not Path(self.index_path).exists():
                raise FileNotFoundError(f"FAISS 索引不存在: {self.index_path}")
            self.index = faiss.read_index(self.index_path)

            if Path(self.metadata_path).exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    for line in f:
                        self.metadata.append(json.loads(line))

            # Demo hash vectors are not compatible with bge-m3 embeddings.
            self.demo_bm25_only = any(
                doc.get("embedding_mode") == "demo-bm25" for doc in self.metadata
            )
            
            tokenized_corpus = [jieba.lcut(doc.get("text", "")) for doc in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"知识库加载完成")

        except Exception as e:
            self.load_error = str(e)
            print(f"知识库加载失败: {e}")

    def _resolve_image_absolute(self, img_path: str) -> Optional[Path]:
        if not img_path: return None
        p = Path(img_path)
        if p.is_absolute() and p.exists(): return p
        filename = p.name
        for source_dir in self.image_source_dirs:
            candidate = Path(source_dir) / filename
            if candidate.exists():
                return candidate
        return None

    def _is_useful_image(self, img_path: str) -> bool:
        if not img_path: return False
        name_lower = Path(img_path).name.lower()
        bad_keywords = ["icon", "logo", "watermark", "avatar", "bg", "background"]
        if any(k in name_lower for k in bad_keywords): return False
        abs_p = self._resolve_image_absolute(img_path)
        if abs_p is None: return False
        try:
            size_kb = abs_p.stat().st_size / 1024
            if size_kb < 10: return False
            return True
        except Exception:
            return False

    def _get_query_embedding(self, text: str) -> Optional[np.ndarray]:
        if self.demo_bm25_only or ollama is None:
            return None
        try:
            resp = ollama.embeddings(model=EMB_MODEL, prompt=text)
            return np.array(resp["embedding"], dtype=np.float32).reshape(1, -1)
        except Exception as e:
            print(f"Embedding 不可用，将仅使用 BM25: {e}")
            return None

    def _vector_search(self, query_emb: Optional[np.ndarray]) -> Tuple[List[float], List[int]]:
        if query_emb is None:
            return [], []
        distances, idxs = self.index.search(query_emb, TOP_K)
        return distances[0].tolist(), idxs[0].tolist()

    def _kw_search(self, query: str) -> Tuple[List[float], List[int]]:
        tokenized_query = jieba.lcut(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n_idxs = np.argsort(doc_scores)[-BM25_TOP_K:][::-1]
        top_n_scores = [doc_scores[i] for i in top_n_idxs]
        return top_n_scores, top_n_idxs.tolist()

    def _rrf_fusion(self, vector_results, keyword_results) -> List[Tuple[int, float]]:
        rrf_scores = {}
        k = RRF_K
        for rank, idx in enumerate(vector_results[1]):
            if idx == -1: continue
            if idx not in rrf_scores: rrf_scores[idx] = 0
            rrf_scores[idx] += 1 / (k + rank + 1)
        for rank, idx in enumerate(keyword_results[1]):
            if idx == -1: continue
            if idx not in rrf_scores: rrf_scores[idx] = 0
            rrf_scores[idx] += 1 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates or self.reranker is None:
            return candidates

        pairs = [[query, doc.get("text", "")[:800]] for doc in candidates]

        scores = self.reranker.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]

        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = score
        
        return sorted(candidates, key=lambda x: x.get('rerank_score', -999), reverse=True)

    # 【超级核心修改】双重过滤逻辑
    def _build_context(self, ranked_docs: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        parts = []
        refs = []
        seen_images = set()

        filtered_docs = []
        is_using_reranker = self.reranker is not None and len(ranked_docs) > 0 and 'rerank_score' in ranked_docs[0]

        # 1. 确定基准分数 (Top 1 的分数)
        top_score = ranked_docs[0].get('rerank_score', 0) if is_using_reranker else 0

        for doc in ranked_docs:
            if is_using_reranker:
                current_score = doc.get('rerank_score', -99)
                
                # 规则1: 绝对分太低，丢弃
                if current_score < RERANK_THRESHOLD:
                    continue
                
                # 规则2: 【分差截断】如果当前分比第一名低了 3.5 分，认为是相似但错误的型号，丢弃！
                # 例: 0808型号(7.2分)，0780型号(3.1分)，差4.1分 > 3.5，触发截断。
                if (top_score - current_score) > RERANK_GAP_THRESHOLD:
                    continue
            
            filtered_docs.append(doc)
        
        # 专业库在低置信时进入无引用直答；其他库至少保留第一条候选。
        if not filtered_docs and ranked_docs:
            final_docs = [] if self.enable_fallback else [ranked_docs[0]]
        else:
            final_docs = filtered_docs[:FINAL_TOP_K]

        for rank, m in enumerate(final_docs, start=1):
            raw_img_paths = m.get("image_paths", []) or []
            filtered_paths = [p for p in raw_img_paths if self._is_useful_image(p)]
            img_web_paths = [f"/img/{Path(p).name}" for p in filtered_paths[:MAX_IMGS_PER_CHUNK]]
            
            unique_context_imgs = []
            for img_path in img_web_paths:
                if img_path not in seen_images:
                    seen_images.add(img_path)
                    unique_context_imgs.append(img_path)

            display_score = m.get("rerank_score", m.get("rrf_score", 0))

            ref = {
                "ref_id": rank,
                "file_path": m.get("file_path"),
                "doc_name": Path(m.get("file_path", "")).name if m.get("file_path") else "",
                "preview_text": (m.get("text", "") or "")[:100],
                "image_paths": img_web_paths,
                "score": float(display_score),
                "type": m.get("type", "text")
            }
            refs.append(ref)

            text_content = m.get("text", "")
            if unique_context_imgs:
                img_str = ", ".join(unique_context_imgs)
                text_content += f"\n\n[相关图片路径: {img_str}]"

            chunk_xml = (
                f'<chunk fileId="{m.get("chunk_id")}" '
                f'fileName="{Path(m.get("file_path", "")).name}" '
                f'score="{display_score:.4f}">'
                f'{text_content}'
                f'</chunk>'
            )
            parts.append(chunk_xml)

        if not parts:
            ctx = "<retrieved_chunks>No relevant information found.</retrieved_chunks>"
        else:
            ctx_content = "\n".join(parts)
            ctx = f"<retrieved_chunks>\n{ctx_content}\n</retrieved_chunks>"
        
        if len(ctx) > CONTEXT_MAX_CHARS:
            ctx = ctx[:CONTEXT_MAX_CHARS] + "\n<!-- Context truncated -->"

        return ctx, refs

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        if not DEEPSEEK_API_KEY:
            return "模型服务未配置：请设置 DEEPSEEK_API_KEY。"

        try:
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": API_GENERATION_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            detail = ""
            if getattr(e, "response", None) is not None:
                detail = e.response.text[:300]
            print(f"DeepSeek API 调用失败: {e} {detail}")
            return f"模型服务连接失败：{detail or str(e)}"
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"DeepSeek API 返回格式异常: {e}")
            return "模型服务返回格式异常。"

    def _ask_llm(self, context: str, question: str) -> str:
        system_prompt = """You are a helpful assistant good at answering questions based on the provided knowledge base.
<instruction>
- Use the retrieved chunks closely.
- Answer in Chinese.
- If chunks contain "[相关图片路径: ...]", strictly list them at the end of your answer as:
  ### 相关图片
  ![](/img/xxxx.png)
- Use Markdown format.
</instruction>"""

        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]

        return self._call_llm(messages)

    def _ask_llm_fallback(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业的中文问答助手。知识库没有返回可用的支撑内容，请直接回答用户问题。"
                    "如果问题不确定或需要非常具体的事实，请明确说明不确定。"
                    "禁止伪造引用、检索片段、文件来源或图片。"
                ),
            },
            {"role": "user", "content": question},
        ]
        return self._call_llm(messages)

    def _generate_suggestions(self, context: str, question: str) -> List[str]:
        short_ctx = context[:800] if context else ""
        prompt = (
            f"基于以下资料和问题，生成3个简短的用户可能感兴趣的后续提问。\n"
            f"资料: {short_ctx}...\n问题: {question}\n"
            f"要求: 只输出3行问题，不要序号。"
        )
        content = self._call_llm([{'role': 'user', 'content': prompt}])
        if content.startswith("模型服务"):
            return ["详细介绍一下？", "有什么特点？", "应用场景？"]
        return [re.sub(r'^[\d\.\-\s]+', '', line.strip()) for line in content.split('\n') if len(line) > 4][:3]

    def chat(self, question: str) -> Dict[str, Any]:
        if self.load_error:
            if self.enable_fallback:
                return {
                    "answer": self._ask_llm_fallback(question),
                    "references": [],
                    "related_questions": self._generate_suggestions("", question),
                }
            return {"answer": f"错误: {self.load_error}", "references": [], "related_questions": []}
        if self.index is None:
            if self.enable_fallback:
                return {
                    "answer": self._ask_llm_fallback(question),
                    "references": [],
                    "related_questions": self._generate_suggestions("", question),
                }
            return {"answer": "索引未初始化", "references": [], "related_questions": []}

        q_emb = self._get_query_embedding(question)
        vec_dists, vec_idxs = self._vector_search(q_emb)
        bm25_scores, bm25_idxs = self._kw_search(question)
        
        rrf_ranked = self._rrf_fusion((vec_dists, vec_idxs), (bm25_scores, bm25_idxs))
        
        candidate_docs = []
        for idx, rrf_score in rrf_ranked[:RRF_CANDIDATES]:
            if idx < 0 or idx >= len(self.metadata): continue
            doc_item = self.metadata[idx].copy()
            doc_item['rrf_score'] = rrf_score 
            candidate_docs.append(doc_item)
            
        final_ranked_docs = self._rerank_candidates(question, candidate_docs)
        
        # 构建上下文时，会自动执行“分差截断”
        context, refs = self._build_context(final_ranked_docs)
        
        if self.enable_fallback and not refs:
            answer_text = self._ask_llm_fallback(question)
        else:
            answer_text = self._ask_llm(context, question)
        related_qs = self._generate_suggestions(context, question)

        return {
            "answer": answer_text,
            "references": refs,
            "related_questions": related_qs
        }
