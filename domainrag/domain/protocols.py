"""检索、模型和知识工具的稳定契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import EvaluationCase, SearchResult

class Retriever(Protocol):
    """检索器协议：输入问题和 top_k，输出按相关性排序的 SearchResult。

    Protocol 是结构化类型约束：类不必显式继承 Retriever，只要实现同样的 search()
    就能被 Hybrid、Reranker 和评测代码使用。这也是以后接向量数据库的扩展点。
    """

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


class ChatModel(Protocol):
    model: str

    def complete(self, messages: list[dict], *, allow_tools: bool, timeout: float) -> dict: ...


@dataclass(frozen=True)
class ApiCallStats:
    """一次模型 API 调用的耗时与 Token 用量，供成本和性能分析使用。"""

    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    """生成文本及其 API 统计，避免只返回字符串后丢失运行证据。"""

    text: str
    stats: ApiCallStats = ApiCallStats()


class AnswerGenerator(Protocol):
    """答案生成器接口；DeepSeek、Ollama 和测试替身都可实现它。"""

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class CitationAssessment:
    """Judge 对某个有效引用编号的 Chunk 级支持判断。"""

    index: int
    supported: bool
    reason: str


@dataclass(frozen=True)
class SemanticJudgement:
    """LLM Judge 返回的语义评分、理由、逐引用判断和 API 统计。"""

    correctness: int
    faithfulness: int
    reason: str
    citation_assessments: tuple[CitationAssessment, ...] | None = None
    stats: ApiCallStats = ApiCallStats()
    # None 仅兼容旧的自定义 Judge；不应把未测量的拒答率记为零。
    abstained: bool | None = None


class AnswerJudge(Protocol):
    """答案裁判接口，用于将具体模型调用与评测主流程解耦。"""

    def judge(
        self,
        case: EvaluationCase,
        answer: str,
        results: list[SearchResult],
    ) -> SemanticJudgement: ...


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索知识库证据，返回来源ID、块ID、规范化文本位置与分数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "按检索返回的来源ID读取规范化文档；start为零基字符偏移。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "minLength": 1},
                    "start": {"type": "integer", "minimum": 0, "default": 0},
                    "max_chars": {"type": "integer", "minimum": 1, "maximum": 4000, "default": 2000},
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
]
