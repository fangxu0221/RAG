"""工具边界、证据一致性、追踪与真实BM25集成测试；不调用模型API。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from domainrag.chunking import chunk_documents
from domainrag.cli import build_retriever
from domainrag.infrastructure.retrievers.bm25 import BM25Retriever
from domainrag.models import Document
from domainrag.text import normalize_text
from domainrag.tools import KnowledgeTools, ToolInputError, ToolSession


class KnowledgeToolTests(unittest.TestCase):
    def setUp(self):
        self.documents = [Document("attention.md", "Attention\n\n显存与序列长度平方有关。"),
                          Document("optimization.md", "梯度裁剪用于限制梯度。")]
        self.chunks = chunk_documents(self.documents, 12, 3)
        self.tools = KnowledgeTools(BM25Retriever(self.chunks), self.documents, {"retriever": "bm25"})

    def test_search_and_read_share_normalized_coordinates(self):
        result = self.tools.search_knowledge("Attention", 2)
        self.assertGreater(result["count"], 0)
        for hit in result["results"]:
            read = self.tools.read_document(hit["source"], hit["start"], hit["end"] - hit["start"])
            self.assertEqual(read["text"].strip(), hit["text"])
        self.assertFalse(result["score_is_probability"])

    def test_read_pagination_without_gaps(self):
        parts, start = [], 0
        while start is not None:
            page = self.tools.read_document("attention.md", start, 5)
            parts.append(page["text"])
            start = page["next_start"]
        self.assertEqual("".join(parts), normalize_text(self.documents[0].text))
        end = self.tools.read_document("attention.md", page["total_chars"])
        self.assertEqual(end["text"], "")
        self.assertIsNone(end["next_start"])

    def test_unknown_and_path_sources_rejected_without_io(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected disk read")):
            for source in ["../secret.txt", "C:/secret.txt", "missing.md", "attention.md/../secret", None]:
                with self.subTest(source=source), self.assertRaises(ToolInputError):
                    self.tools.read_document(source)

    def test_numeric_and_query_boundaries(self):
        for value in [True, 0, -1, 11, 1.5, "3"]:
            with self.subTest(value=value), self.assertRaises(ToolInputError):
                self.tools.search_knowledge("Attention", value)
        for query in ["", "   ", None, "x" * 2001]:
            with self.assertRaises(ToolInputError):
                self.tools.search_knowledge(query)
        for args in [{"start": -1}, {"start": True}, {"start": 10000},
                     {"max_chars": 0}, {"max_chars": 4001}]:
            with self.assertRaises(ToolInputError):
                self.tools.read_document("attention.md", **args)

    def test_trace_success_error_and_request_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ToolSession(self.tools, Path(tmp)) as session:
                hit = session.call("search_knowledge", {"query": "Attention", "top_k": 1})
                source = hit["result"]["results"][0]["source"]
                session.call("read_document", {"source": source})
                invalid = session.call("read_document", {"source": "../secret.txt"})
                self.assertEqual(invalid["error"]["code"], "invalid_arguments")
            rows = [json.loads(line) for line in session.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["request_id"] for row in rows}, {session.request_id})
            self.assertEqual(rows[0]["config"], {"retriever": "bm25"})
            self.assertEqual(rows[-1]["failed_calls"], 1)
            self.assertEqual(rows[-1]["status"], "completed_with_errors")
            self.assertIsNone(rows[-1]["token_usage"])
            original = session.path.read_bytes()
            with ToolSession(self.tools, Path(tmp)) as other:
                self.assertNotEqual(other.path, session.path)
            self.assertEqual(session.path.read_bytes(), original)
            with self.assertRaises(RuntimeError):
                session.call("search_knowledge", {"query": "late"})

    def test_dispatch_validation_and_backend_failure(self):
        with tempfile.TemporaryDirectory() as tmp, ToolSession(self.tools, Path(tmp)) as session:
            for name, args in [("unknown", {}), ("search_knowledge", {}),
                               ("search_knowledge", {"query": "x", "extra": 1})]:
                self.assertEqual(session.call(name, args)["error"]["code"], "invalid_arguments")
            with patch.object(self.tools._retriever, "search", side_effect=RuntimeError("secret")):
                result = session.call("search_knowledge", {"query": "x"})
                self.assertEqual(result["error"]["code"], "tool_failed")
                self.assertNotIn("secret", json.dumps(result))
            self.assertNotIn("secret", session.path.read_text(encoding="utf-8"))

    def test_aborted_session_records_terminal_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                with ToolSession(self.tools, Path(tmp)) as session:
                    raise RuntimeError("caller failed")
            rows = session.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(rows[-1])["status"], "aborted")

    def test_empty_results_are_successful(self):
        with patch.object(self.tools._retriever, "search", return_value=[]):
            self.assertEqual(self.tools.search_knowledge("missing")["results"], [])

    def test_factory_snapshot_and_existing_cli_agree(self):
        config = Path(__file__).resolve().parents[1] / "configs/bm25_baseline.json"
        original, _ = build_retriever(config)
        tools = KnowledgeTools.from_config(config)
        actual = tools.search_knowledge("为什么注意力分数要除以根号 d_k？", 3)
        expected = original.search("为什么注意力分数要除以根号 d_k？", 3)
        self.assertEqual([(r["chunk_id"], r["score"]) for r in actual["results"]],
                         [(r.chunk.chunk_id, r.score) for r in expected])

    def test_factory_checks_frozen_data_before_building(self):
        config = Path(__file__).resolve().parents[1] / "configs/bm25_baseline.json"
        with patch("domainrag.bootstrap.verify_frozen_dataset", side_effect=ValueError("tampered")) as verify, \
                patch("domainrag.bootstrap.load_documents") as load, \
                patch("domainrag.bootstrap.build_retriever_from_documents") as build:
            with self.assertRaisesRegex(ValueError, "tampered"):
                KnowledgeTools.from_config(config)
            verify.assert_called_once_with(config)
            load.assert_not_called()
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
