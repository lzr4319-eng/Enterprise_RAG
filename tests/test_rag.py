"""Offline checks: no enterprise data or external model requests are used."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "main"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import rag_core
from build_demo_indexes import build_demo_indexes
from rag_core import RAG


class DemoIndexTests(unittest.TestCase):
    def test_demo_indexes_are_marked_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            build_demo_indexes(directory)
            for category in ("employee", "product"):
                base = Path(directory) / category
                index = faiss.read_index(str(base / "faiss_index.index"))
                records = [
                    json.loads(line)
                    for line in (base / "faiss_index_metadata.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(index.ntotal, len(records))
                self.assertEqual(index.d, 1024)
                self.assertTrue(all(r["embedding_mode"] == "demo-bm25" for r in records))
            original = (Path(directory) / "employee" / "faiss_index.index").read_bytes()
            with self.assertRaises(FileExistsError):
                build_demo_indexes(directory)
            self.assertEqual(
                original, (Path(directory) / "employee" / "faiss_index.index").read_bytes()
            )

    def test_demo_engine_skips_ollama_and_returns_references(self):
        with tempfile.TemporaryDirectory() as directory:
            build_demo_indexes(directory)
            base = Path(directory) / "employee"
            with patch.object(rag_core, "RERANK_MODEL_NAME", ""):
                engine = RAG(str(base / "faiss_index.index"),
                             str(base / "faiss_index_metadata.jsonl"))
            self.assertIsNone(engine.load_error)
            with patch.object(rag_core, "ollama") as ollama:
                self.assertIsNone(engine._get_query_embedding("新员工入职"))
                ollama.embeddings.assert_not_called()
            with patch.object(engine, "_ask_llm", return_value="演示回答"), \
                 patch.object(engine, "_generate_suggestions", return_value=[]):
                result = engine.chat("新员工入职")
            self.assertEqual(result["answer"], "演示回答")
            self.assertTrue(result["references"])


class FallbackTests(unittest.TestCase):
    def make_engine(self, directory, enabled):
        with patch.object(rag_core, "RERANK_MODEL_NAME", ""):
            return RAG(str(Path(directory) / "missing.index"),
                       str(Path(directory) / "missing.jsonl"), enable_fallback=enabled)

    def test_missing_index_uses_uncited_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, True)
            with patch.object(engine, "_ask_llm_fallback", return_value="专业回答") as fallback, \
                 patch.object(engine, "_generate_suggestions", return_value=[]), \
                 patch.object(engine, "_ask_llm") as grounded:
                result = engine.chat("什么是光刻？")
            fallback.assert_called_once_with("什么是光刻？")
            grounded.assert_not_called()
            self.assertEqual(result["references"], [])

    def test_missing_index_without_fallback_does_not_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, False)
            with patch.object(engine, "_call_llm") as generate:
                result = engine.chat("入职流程？")
            generate.assert_not_called()
            self.assertEqual(result["references"], [])
            self.assertIn("错误", result["answer"])

    def test_filtered_candidates_do_not_become_professional_references(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, True)
            engine.reranker = Mock()
            docs = [{"text": "无关资料", "file_path": "demo.md", "rerank_score": -3.0}]
            _, references = engine._build_context(docs)
            self.assertEqual(references, [])
            engine.enable_fallback = False
            _, references = engine._build_context(docs)
            self.assertEqual(len(references), 1)

    def test_absolute_and_relative_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, True)
            engine.reranker = Mock()
            docs = [
                {"text": "相关资料", "file_path": "top.md", "rerank_score": 4.0},
                {"text": "分差过大", "file_path": "gap.md", "rerank_score": 0.2},
                {"text": "分数过低", "file_path": "low.md", "rerank_score": -3.0},
            ]
            _, references = engine._build_context(docs)
            self.assertEqual([r["doc_name"] for r in references], ["top.md"])

    def test_empty_retrieval_routes_to_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, True)
            engine.load_error = None
            engine.index = Mock()
            with patch.object(engine, "_get_query_embedding", return_value=None), \
                 patch.object(engine, "_kw_search", return_value=([], [])), \
                 patch.object(engine, "_ask_llm_fallback", return_value="直答") as fallback, \
                 patch.object(engine, "_ask_llm") as grounded, \
                 patch.object(engine, "_generate_suggestions", return_value=[]):
                result = engine.chat("专业问题")
            fallback.assert_called_once()
            grounded.assert_not_called()
            self.assertEqual(result["references"], [])

    def test_missing_api_key_does_not_send_request(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, True)
            with patch.object(rag_core, "DEEPSEEK_API_KEY", ""), \
                 patch.object(rag_core.requests, "post") as request:
                answer = engine._call_llm([])
            request.assert_not_called()
            self.assertIn("DEEPSEEK_API_KEY", answer)


if __name__ == "__main__":
    unittest.main()
