"""核心回归测试。

这些测试使用很小的可控样本和 Fake 模型，验证公式、数据流与边界条件；它们不依赖
网络或真实大模型。阅读时重点看“构造什么输入 -> 预期什么排序/指标”，不用记 Mock 语法。
"""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from domainrag.application.evaluation_service import evaluate_answers, evaluate_citations
from domainrag.chunking import chunk_document
from domainrag.domain.protocols import ApiCallStats, CitationAssessment, GenerationResult, SemanticJudgement
from domainrag.evaluation import evaluate_retrieval
from domainrag.infrastructure.llm.answer_judge import DeepSeekAnswerJudge
from domainrag.infrastructure.llm.generator import DeepSeekGenerator
from domainrag.infrastructure.retrievers.bm25 import BM25Retriever
from domainrag.infrastructure.retrievers.embedding import EmbeddingRetriever
from domainrag.infrastructure.retrievers.hybrid import HybridRetriever
from domainrag.infrastructure.retrievers.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from domainrag.models import Chunk, Document, EvaluationCase, SearchResult
from domainrag.text import tokenize


class TokenizationTests(unittest.TestCase):
    """确认中英文混合文本能产生 BM25 所需的关键词 token。"""

    def test_mixed_text_tokenization(self) -> None:
        tokens = tokenize("BM25适合关键词检索")
        self.assertIn("bm25", tokens)
        self.assertIn("关键", tokens)
        self.assertIn("检索", tokens)


class ChunkingTests(unittest.TestCase):
    """确认滑动窗口的重叠长度、稳定 ID 和参数校验。"""

    def test_chunk_overlap_and_ids(self) -> None:
        document = Document(source="sample.md", text="甲" * 25)
        chunks = chunk_document(document, chunk_size=10, chunk_overlap=2)
        self.assertEqual([chunk.chunk_id for chunk in chunks], [
            "sample.md#0000",
            "sample.md#0001",
            "sample.md#0002",
        ])
        self.assertEqual(chunks[0].end - chunks[1].start, 2)

    def test_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_document(Document("x.md", "text"), 10, 10)


class RetrievalTests(unittest.TestCase):
    """用两篇极小文档验证 BM25 排序及检索指标计算。"""

    def setUp(self) -> None:
        documents = [
            Document("attention.md", "注意力分数除以根号维度可以避免 softmax 饱和"),
            Document("rag.md", "BM25 使用关键词检索，稠密检索使用语义向量"),
        ]
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document, chunk_size=100, chunk_overlap=0)
        ]
        self.retriever = BM25Retriever(chunks)

    def test_relevant_document_ranks_first(self) -> None:
        results = self.retriever.search("BM25 关键词", top_k=2)
        self.assertEqual(results[0].chunk.source, "rag.md")

    def test_evaluation_metrics(self) -> None:
        cases = [
            EvaluationCase("q1", "BM25 关键词", ("rag.md",)),
            EvaluationCase("q2", "softmax 饱和", ("attention.md",)),
        ]
        report = evaluate_retrieval(self.retriever, cases, top_k=1)
        self.assertEqual(report["summary"]["hit_rate@1"], 1.0)
        self.assertEqual(report["summary"]["mrr"], 1.0)


class _FakeEncoder:
    """把注意力语义映射到 [1,0]，其他语义映射到 [0,1] 的可控假模型。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs) -> list[list[float]]:
        self.calls.append(sentences)
        vectors = []
        for sentence in sentences:
            if "注意力" in sentence or "聚焦" in sentence:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class EmbeddingRetrievalTests(unittest.TestCase):
    """验证向量相似度排序，以及检索指令只加在 query 而不加在 Chunk 上。"""

    def setUp(self) -> None:
        self.encoder = _FakeEncoder()
        self.chunks = [
            Chunk("attention.md#0000", "attention.md", "注意力机制计算相关性", 0, 10),
            Chunk("rag.md#0000", "rag.md", "BM25 根据关键词匹配文档", 0, 14),
        ]
        self.retriever = EmbeddingRetriever(
            self.chunks,
            query_instruction="检索：",
            encoder=self.encoder,
        )

    def test_semantic_vector_ranks_relevant_chunk_first(self) -> None:
        results = self.retriever.search("模型如何聚焦重要信息", top_k=2)
        self.assertEqual(results[0].chunk.source, "attention.md")
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_query_instruction_is_only_added_to_query(self) -> None:
        self.retriever.search("模型如何聚焦重要信息", top_k=1)
        self.assertNotIn("检索：", self.encoder.calls[0][0])
        self.assertEqual(self.encoder.calls[1], ["检索：模型如何聚焦重要信息"])

    def test_top_k_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.search("问题", top_k=0)


class _StaticRetriever:
    """按预设顺序返回 Chunk，并记录上游实际请求了多少候选。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.requested_top_k: list[int] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.requested_top_k.append(top_k)
        return [
            SearchResult(chunk=chunk, score=float(len(self.chunks) - index))
            for index, chunk in enumerate(self.chunks[:top_k])
        ]


class HybridRetrievalTests(unittest.TestCase):
    """验证 RRF 共识加分、候选池大小和非法配置。"""

    def setUp(self) -> None:
        self.chunk_a = Chunk("a#0000", "a.md", "A", 0, 1)
        self.chunk_b = Chunk("b#0000", "b.md", "B", 0, 1)
        self.chunk_c = Chunk("c#0000", "c.md", "C", 0, 1)
        self.first = _StaticRetriever([self.chunk_a, self.chunk_b, self.chunk_c])
        self.second = _StaticRetriever([self.chunk_b, self.chunk_c, self.chunk_a])

    def test_rrf_rewards_consensus_across_rankings(self) -> None:
        # B 在两路分别排第 2、第 1，累计 RRF 分数后成为最终第 1。
        retriever = HybridRetriever(
            [self.first, self.second], candidate_k=3, rrf_k=60
        )
        results = retriever.search("question", top_k=2)

        self.assertEqual([result.chunk for result in results], [
            self.chunk_b,
            self.chunk_a,
        ])
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(self.first.requested_top_k, [3])
        self.assertEqual(self.second.requested_top_k, [3])

    def test_candidate_pool_is_never_smaller_than_final_top_k(self) -> None:
        retriever = HybridRetriever(
            [self.first, self.second], candidate_k=1, rrf_k=60
        )
        retriever.search("question", top_k=2)
        self.assertEqual(self.first.requested_top_k, [2])

    def test_hybrid_requires_two_retrievers(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetriever([self.first])

    def test_hybrid_parameters_are_validated(self) -> None:
        for invalid_options in ({"candidate_k": 0}, {"rrf_k": -1}):
            with self.subTest(invalid_options=invalid_options):
                with self.assertRaises(ValueError):
                    HybridRetriever(
                        [self.first, self.second],
                        **invalid_options,
                    )


class _FakePairScorer:
    """返回预设 Cross-Encoder 分数，以隔离真实模型并专门测试重排逻辑。"""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs) -> list[float]:
        self.calls.append((inputs, kwargs))
        return self.scores[:len(inputs)]


class CrossEncoderRerankerTests(unittest.TestCase):
    """验证第一阶段候选会按第二阶段分数重排，并正确传递模型配置。"""

    def setUp(self) -> None:
        self.chunk_a = Chunk("a#0000", "a.md", "文档 A", 0, 4)
        self.chunk_b = Chunk("b#0000", "b.md", "文档 B", 0, 4)
        self.chunk_c = Chunk("c#0000", "c.md", "文档 C", 0, 4)
        self.first_stage = _StaticRetriever([
            self.chunk_a,
            self.chunk_b,
            self.chunk_c,
        ])
        self.scorer = _FakePairScorer([0.1, 0.9, 0.2])

    def test_cross_encoder_reorders_first_stage_candidates(self) -> None:
        reranker = CrossEncoderReranker(
            self.first_stage,
            candidate_k=3,
            batch_size=2,
            scorer=self.scorer,
        )
        results = reranker.search("问题", top_k=2)

        self.assertEqual([result.chunk for result in results], [
            self.chunk_b,
            self.chunk_c,
        ])
        self.assertEqual(self.first_stage.requested_top_k, [3])
        pairs, options = self.scorer.calls[0]
        self.assertEqual(pairs, [
            ("问题", "文档 A"),
            ("问题", "文档 B"),
            ("问题", "文档 C"),
        ])
        self.assertEqual(options["batch_size"], 2)
        self.assertFalse(options["show_progress_bar"])

    def test_reranker_candidate_pool_covers_final_top_k(self) -> None:
        reranker = CrossEncoderReranker(
            self.first_stage,
            candidate_k=1,
            scorer=self.scorer,
        )
        reranker.search("问题", top_k=2)
        self.assertEqual(self.first_stage.requested_top_k, [2])

    def test_torch_backend_does_not_select_an_onnx_file(self) -> None:
        reranker = CrossEncoderReranker(
            self.first_stage,
            backend="torch",
            scorer=self.scorer,
        )
        self.assertIsNone(reranker.model_file)

    def test_reranker_passes_configured_backend_to_model(self) -> None:
        # 保留已有离线测试修复：注入可选模块替身，不下载模型或改变重排实现。
        mocked_model = Mock(return_value=self.scorer)
        module = ModuleType("sentence_transformers")
        module.CrossEncoder = mocked_model
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            reranker = CrossEncoderReranker(
                self.first_stage,
                backend="onnx",
                model_file="onnx/custom_quantized.onnx",
            )

        mocked_model.assert_called_once_with(
            DEFAULT_RERANKER_MODEL,
            max_length=512,
            backend="onnx",
            model_kwargs={"file_name": "onnx/custom_quantized.onnx"},
        )
        self.assertEqual(reranker.model_file, "onnx/custom_quantized.onnx")

    def test_reranker_parameters_are_validated(self) -> None:
        for invalid_options in (
            {"candidate_k": 0},
            {"batch_size": 0},
            {"max_length": 0},
            {"backend": "invalid"},
        ):
            with self.subTest(invalid_options=invalid_options):
                with self.assertRaises(ValueError):
                    CrossEncoderReranker(
                        self.first_stage,
                        scorer=self.scorer,
                        **invalid_options,
                    )


class _FakeAnswerGenerator:
    """返回固定答案与统计，证明答案评测主流程没有隐藏的真实 API 调用。"""

    def __init__(self, answer: str, stats: ApiCallStats | None = None) -> None:
        self.answer = answer
        self.stats = stats or ApiCallStats()
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        self.calls.append((question, results))
        return GenerationResult(self.answer, self.stats)


class _FakeAnswerJudge:
    """返回固定 Judge 结果，并记录是否复用了生成器看到的检索结果。"""

    def __init__(self, judgement: SemanticJudgement) -> None:
        self.judgement = judgement
        self.calls: list[tuple[EvaluationCase, str, list[SearchResult]]] = []

    def judge(
        self,
        case: EvaluationCase,
        answer: str,
        results: list[SearchResult],
    ) -> SemanticJudgement:
        self.calls.append((case, answer, results))
        return self.judgement


class AnswerEvaluationTests(unittest.TestCase):
    """验证引用指标、端到端汇总、失败筛选和 DeepSeek Judge JSON 解析。"""

    def setUp(self) -> None:
        self.rag_chunk = Chunk(
            "rag.md#0000",
            "rag.md",
            "RAG 先检索资料再生成回答。",
            0,
            16,
        )
        self.attention_chunk = Chunk(
            "attention.md#0000",
            "attention.md",
            "注意力根据相关性聚合信息。",
            0,
            14,
        )
        self.results = [
            SearchResult(self.rag_chunk, 2.0),
            SearchResult(self.attention_chunk, 1.0),
        ]

    def test_citation_metrics_penalize_invalid_indices(self) -> None:
        metrics = evaluate_citations(
            "RAG 会先检索。[来源：1] 另一个说法。[3]",
            self.results,
            ("rag.md",),
        )

        self.assertEqual(metrics["cited_indices"], [1, 3])
        self.assertEqual(metrics["invalid_indices"], [3])
        self.assertEqual(metrics["citation_validity"], 0.5)
        self.assertEqual(metrics["citation_precision"], 1.0)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertEqual(metrics["citation_accuracy"], 0.5)

    def test_chunk_support_penalizes_irrelevant_chunk_from_same_source(self) -> None:
        # 两个 Chunk 虽来自同一文档，但第二块不支持答案，Precision 应受到惩罚。
        irrelevant_same_source = Chunk(
            "rag.md#0001",
            "rag.md",
            "这段只讨论无关的部署配置。",
            17,
            31,
        )
        results = [self.results[0], SearchResult(irrelevant_same_source, 1.5)]

        metrics = evaluate_citations(
            "RAG 会先检索再生成。[1][2]",
            results,
            ("rag.md",),
            (
                CitationAssessment(1, True, "直接支持回答"),
                CitationAssessment(2, False, "与回答事实无关"),
            ),
        )

        self.assertEqual(metrics["supported_indices"], [1])
        self.assertEqual(metrics["unsupported_indices"], [2])
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertAlmostEqual(metrics["citation_accuracy"], 2 / 3)

    def test_answer_evaluation_reuses_retrieved_results(self) -> None:
        # 生成器和 Judge 必须看到同一 list 对象，避免重复检索产生不同上下文编号。
        retriever = _StaticRetriever([self.rag_chunk, self.attention_chunk])
        generator = _FakeAnswerGenerator(
            "RAG 先检索再生成。[1]",
            ApiCallStats(0.4, 100, 20, 120),
        )
        judge = _FakeAnswerJudge(
            SemanticJudgement(
                2,
                1,
                "结论正确",
                (CitationAssessment(1, True, "直接支持回答"),),
                ApiCallStats(0.6, 80, 10, 90),
            )
        )
        case = EvaluationCase(
            "rag-answer-001",
            "RAG 是怎样回答问题的？",
            ("rag.md",),
            expected_answer="先检索相关资料，再生成回答。",
            metadata={"question_type": "direct"},
        )

        report = evaluate_answers(
            retriever,
            [case],
            generator,
            judge,
            top_k=2,
        )

        self.assertEqual(retriever.requested_top_k, [2])
        self.assertIs(generator.calls[0][1], judge.calls[0][2])
        self.assertEqual(report["summary"]["correctness"], 1.0)
        self.assertEqual(report["summary"]["faithfulness"], 0.5)
        self.assertEqual(report["summary"]["citation_accuracy"], 1.0)
        self.assertEqual(report["summary"]["api"]["calls"], 2)
        self.assertEqual(report["summary"]["api"]["total_tokens"], 210)
        self.assertEqual(report["by_question_type"]["direct"]["cases"], 1)
        self.assertEqual(report["failure_cases"][0]["id"], "rag-answer-001")
        self.assertEqual(report["cases"][0]["metadata"]["question_type"], "direct")

    def test_missing_expected_answer_fails_before_generation(self) -> None:
        retriever = _StaticRetriever([self.rag_chunk])
        generator = _FakeAnswerGenerator("不应该被调用")
        judge = _FakeAnswerJudge(SemanticJudgement(2, 2, ""))
        case = EvaluationCase("missing", "问题", ("rag.md",))

        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_answers(retriever, [case], generator, judge, top_k=1)

        self.assertEqual(generator.calls, [])
        self.assertEqual(retriever.requested_top_k, [])

    @patch("domainrag.infrastructure.llm.answer_judge.urlopen")
    def test_deepseek_judge_requests_and_parses_json(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "correctness": 2,
                                    "faithfulness": 1,
                                    "reason": "核心答案正确",
                                    "abstained": False,
                                    "citation_assessments": [
                                        {
                                            "index": 1,
                                            "supported": True,
                                            "reason": "资料直接支持",
                                        }
                                    ],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 90,
                    "completion_tokens": 20,
                    "total_tokens": 110,
                },
            }
        )
        case = EvaluationCase(
            "q1",
            "RAG 是什么？",
            ("rag.md",),
            expected_answer="检索增强生成。",
        )
        judge = DeepSeekAnswerJudge(api_key="test-key")

        judgement = judge.judge(case, "RAG 是检索增强生成。[1]", self.results)

        self.assertEqual(judgement.correctness, 2)
        self.assertEqual(judgement.faithfulness, 1)
        self.assertEqual(judgement.reason, "核心答案正确")
        self.assertEqual(
            judgement.citation_assessments,
            (CitationAssessment(1, True, "资料直接支持"),),
        )
        self.assertEqual(judgement.stats.total_tokens, 110)
        request = mocked_urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["temperature"], 0)
        self.assertIn("参考答案", request_body["messages"][1]["content"])


class _FakeResponse:
    """模拟 urlopen 返回的上下文管理器，把字典编码成 HTTP 响应字节。"""

    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class DeepSeekGeneratorTests(unittest.TestCase):
    """不访问网络，检查 API key、请求地址、请求体和响应统计解析。"""

    def setUp(self) -> None:
        chunk = Chunk("rag.md#0000", "rag.md", "BM25 是关键词检索方法。", 0, 15)
        self.results = [SearchResult(chunk=chunk, score=1.25)]

    def test_api_key_is_required(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                DeepSeekGenerator()

    @patch("domainrag.infrastructure.llm.generator.urlopen")
    def test_generate_uses_chat_completions(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {"message": {"content": "BM25 根据关键词检索。[1]"}}
                ],
                "usage": {
                    "prompt_tokens": 60,
                    "completion_tokens": 12,
                    "total_tokens": 72,
                },
            }
        )
        generator = DeepSeekGenerator(api_key="test-key")

        generation = generator.generate("BM25 是什么？", self.results)

        self.assertEqual(generation.text, "BM25 根据关键词检索。[1]")
        self.assertEqual(generation.stats.total_tokens, 72)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["model"], "deepseek-v4-flash")
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["temperature"], 0)
        self.assertIn("rag.md#0000", request_body["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
