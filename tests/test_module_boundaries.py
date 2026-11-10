"""验证兼容导出与不依赖 CLI 的应用/工作流导入。"""

from importlib import import_module
from pathlib import Path
import subprocess
import sys
import unittest

from domainrag.frozen import runtime_fingerprint, text_sha256


ROOT = Path(__file__).resolve().parents[1]


class ModuleBoundaryTests(unittest.TestCase):
    def test_legacy_exports_use_the_canonical_objects(self):
        pairs = (
            ("cli", "bootstrap", "build_retriever"),
            ("cli", "bootstrap", "build_retriever_from_documents"),
            ("models", "domain", "Document"),
            ("evidence", "domain", "EvidenceGroundTruth"),
            ("retriever", "domain.protocols", "Retriever"),
            ("tools", "application.knowledge_service", "KnowledgeTools"),
            ("tools", "application.knowledge_service", "ToolSession"),
            ("agent", "workflows.native_agent", "run_agent"),
            ("agent", "infrastructure.llm.tool_model", "DeepSeekToolModel"),
            ("bm25", "infrastructure.retrievers.bm25", "BM25Retriever"),
            ("embedding", "infrastructure.retrievers.embedding", "EmbeddingRetriever"),
            ("hybrid", "infrastructure.retrievers.hybrid", "HybridRetriever"),
            ("reranker", "infrastructure.retrievers.reranker", "CrossEncoderReranker"),
            ("generator", "infrastructure.llm.generator", "DeepSeekGenerator"),
            ("answer_evaluation", "application.evaluation_service", "evaluate_answers"),
            ("answer_evaluation", "infrastructure.llm.answer_judge", "DeepSeekAnswerJudge"),
        )
        for old, new, symbol in pairs:
            with self.subTest(symbol=symbol):
                self.assertIs(getattr(import_module(f"domainrag.{old}"), symbol),
                              getattr(import_module(f"domainrag.{new}"), symbol))

    def test_fresh_imports_do_not_pull_in_cli_or_optional_models(self):
        for modules in (
            "domainrag.tools, domainrag.agent, domainrag.bootstrap",
            "domainrag.bootstrap, domainrag.workflows.native_agent, domainrag.tools",
        ):
            with self.subTest(modules=modules):
                process = subprocess.run(
                    [sys.executable, "-B", "-X", "utf8", "-c",
                     f"import sys; import {modules}; "
                     "assert 'domainrag.cli' not in sys.modules; "
                     "assert 'sentence_transformers' not in sys.modules"],
                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15,
                )
                self.assertEqual(process.returncode, 0, process.stderr)

    def test_runtime_fingerprint_includes_moved_implementation(self):
        hashes = runtime_fingerprint()["code_sha256"]
        for name in ("bm25.py", "infrastructure/retrievers/bm25.py", "workflows/native_agent.py"):
            self.assertEqual(hashes[name], text_sha256(ROOT / "domainrag" / name))
