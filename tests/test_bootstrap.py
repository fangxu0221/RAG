"""装配回归，使用真实 BM25 和离线模型替身。"""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from domainrag import bootstrap, cli
from domainrag.infrastructure.retrievers.bm25 import BM25Retriever
from domainrag.infrastructure.retrievers.embedding import EmbeddingRetriever
from domainrag.infrastructure.retrievers.hybrid import HybridRetriever
from domainrag.infrastructure.retrievers.reranker import CrossEncoderReranker
from domainrag.models import Document


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.documents = [Document("a.md", "注意力分数"), Document("b.md", "优化器")]
        self.config = {"chunk_size": 420, "chunk_overlap": 80}
        self.models = ModuleType("sentence_transformers")
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
        scorer = Mock()
        scorer.predict.side_effect = lambda pairs, **kwargs: [float(i) for i, _ in enumerate(pairs)]
        self.models.SentenceTransformer = Mock(return_value=encoder)
        self.models.CrossEncoder = Mock(return_value=scorer)

    def test_bm25_uses_existing_snapshot_and_config(self):
        cfg = {**self.config, "bm25": {"k1": 1.8, "b": 0.4}}
        with patch.object(bootstrap, "load_documents", side_effect=AssertionError("unexpected reload")):
            retriever = bootstrap.build_retriever_from_documents(self.documents, cfg)
        self.assertIsInstance(retriever, BM25Retriever)
        self.assertEqual((retriever.k1, retriever.b), (1.8, 0.4))
        self.assertEqual(retriever.search("注意力", 1)[0].chunk.source, "a.md")

    def test_embedding_receives_configured_model_and_instruction(self):
        cfg = {**self.config, "retriever": "embedding", "embedding": {
            "model": "fixture-embedding", "query_instruction": "query: ", "batch_size": 7,
        }}
        with patch.dict(sys.modules, {"sentence_transformers": self.models}):
            retriever = bootstrap.build_retriever_from_documents(self.documents, cfg)
        self.assertIsInstance(retriever, EmbeddingRetriever)
        self.models.SentenceTransformer.assert_called_once_with("fixture-embedding")
        self.assertEqual((retriever.query_instruction, retriever.batch_size), ("query: ", 7))
        self.assertEqual(len(retriever.search("注意力", 1)), 1)

    def test_embedding_can_use_cached_model_without_network_and_fallback(self):
        hub = ModuleType("huggingface_hub")
        hub.try_to_load_from_cache = Mock(return_value="/cache/modules.json")
        encoder = self.models.SentenceTransformer.return_value
        config = {"retriever": "embedding", **self.config}

        def build_dense(chunks, config):
            return bootstrap.build_embedding(chunks, config, prefer_cached_model=True)

        with patch.dict(sys.modules, {"sentence_transformers": self.models, "huggingface_hub": hub}):
            bootstrap.build_retriever_from_documents(self.documents, config, embedding_builder=build_dense)
            self.models.SentenceTransformer.assert_called_once_with(str(Path("/cache")), local_files_only=True)
            self.models.SentenceTransformer.reset_mock()
            self.models.SentenceTransformer.side_effect = [OSError("incomplete cache"), encoder]
            bootstrap.build_retriever_from_documents(self.documents, config, embedding_builder=build_dense)
            self.assertEqual([call.kwargs for call in self.models.SentenceTransformer.call_args_list],
                             [{"local_files_only": True}, {}])

    def test_hybrid_wires_both_retrievers(self):
        cfg = {**self.config, "retriever": "hybrid", "hybrid": {"candidate_k": 4, "rrf_k": 11}}
        with patch.dict(sys.modules, {"sentence_transformers": self.models}):
            retriever = bootstrap.build_retriever_from_documents(self.documents, cfg)
        self.assertIsInstance(retriever, HybridRetriever)
        self.assertEqual(tuple(map(type, retriever.retrievers)), (BM25Retriever, EmbeddingRetriever))
        self.assertEqual((retriever.candidate_k, retriever.rrf_k), (4, 11))

    def test_reranker_wraps_the_configured_first_stage(self):
        cfg = {**self.config, "retriever": "hybrid", "reranker": {
            "enabled": True, "model": "fixture-reranker", "backend": "onnx",
            "model_file": "onnx/fixture.onnx", "candidate_k": 5, "batch_size": 2, "max_length": 64,
        }}
        with patch.dict(sys.modules, {"sentence_transformers": self.models}):
            retriever = bootstrap.build_retriever_from_documents(self.documents, cfg)
        self.assertIsInstance(retriever, CrossEncoderReranker)
        self.assertIsInstance(retriever.retriever, HybridRetriever)
        self.assertEqual((retriever.candidate_k, retriever.batch_size), (5, 2))
        self.models.CrossEncoder.assert_called_once_with(
            "fixture-reranker", max_length=64, backend="onnx", model_kwargs={"file_name": "onnx/fixture.onnx"}
        )
        self.assertEqual(len(retriever.search("注意力", 1)), 1)

    def test_evidence_cli_uses_one_snapshot_for_index_and_scoring(self):
        config = ROOT / "configs/railway_dev_bm25.json"
        with patch.object(bootstrap, "load_documents", wraps=bootstrap.load_documents) as load, \
                patch.object(cli, "build_retriever_from_documents", wraps=cli.build_retriever_from_documents) as build, \
                patch.object(cli, "evaluate_retrieval", wraps=cli.evaluate_retrieval) as score, \
                patch.object(cli, "write_json"), redirect_stdout(io.StringIO()):
            cli.run_benchmark(argparse.Namespace(config=str(config), output="unused.json"))
        load.assert_called_once()
        self.assertIs(build.call_args.args[0], score.call_args.kwargs["documents"])
