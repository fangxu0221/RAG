"""复用现有检索、生成和 Agent 入口，提供界面需要的请求结果。"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.error import HTTPError, URLError
from uuid import uuid4

from domainrag.application.evaluation_service import CITATION_PATTERN
from domainrag.application.knowledge_service import KnowledgeTools
from domainrag.bootstrap import build_retriever_from_documents, load_retrieval_snapshot
from domainrag.domain.protocols import AnswerGenerator, Retriever
from domainrag.infrastructure.llm.generator import DeepSeekGenerator
from domainrag.infrastructure.llm.tool_model import DeepSeekToolModel
from domainrag.infrastructure.storage.trace import JsonlTrace
from domainrag.models import Document, SearchResult
from domainrag.workflows.native_agent import run_agent


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "results" / "studio_traces"
MODE_LABELS = {"rag": "纯 RAG", "agent": "原生 Agent", "langgraph": "LangGraph V1"}


@dataclass(frozen=True)
class Strategy:
    label: str
    filename: str
    extra: str | None = None


STRATEGIES = {
    "bm25": Strategy("BM25", "railway_dev_bm25.json"),
    "embedding": Strategy("BGE Embedding", "railway_dev_embedding.json", "embedding"),
    "rrf": Strategy("BM25 + BGE + RRF", "railway_dev_hybrid_rrf.json", "embedding"),
    "reranker": Strategy("RRF + CrossEncoder Reranker", "railway_dev_hybrid_reranker.json", "reranker"),
}
DEFAULT_STRATEGY = "reranker"


def config_path(strategy: str) -> Path:
    return ROOT / "configs" / STRATEGIES[strategy].filename


def snapshot(strategy: str, overrides: dict, top_k: int | None) -> tuple[dict, list[Document]]:
    config, documents = load_retrieval_snapshot(config_path(strategy))
    for section, values in overrides.items():
        config[section] = {**config[section], **values}
    if top_k is not None:
        config["top_k"] = top_k
    return config, documents


class SharedRetriever:
    """串行使用同一缓存实例内的编码器与重排器。"""

    def __init__(self, retriever: Retriever):
        self._retriever = retriever
        self._lock = Lock()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        with self._lock:
            return self._retriever.search(query, top_k)


def prepare_retriever(documents: tuple[Document, ...], index_config: dict) -> SharedRetriever:
    return SharedRetriever(build_retriever_from_documents(list(documents), index_config))


def generation_issue(mode: str) -> str | None:
    if mode == "langgraph" and find_spec("langgraph") is None:
        return '缺少 LangGraph，请执行 python -m pip install -e ".[langgraph]"。'
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "当前进程未设置 DEEPSEEK_API_KEY。配置后重新启动界面即可发送问答。"
    return None


def retrieval_issue(strategy: str, config: dict) -> str | None:
    extra = STRATEGIES[strategy].extra
    missing = extra and find_spec("sentence_transformers") is None
    if extra == "reranker" and config["reranker"]["backend"] == "onnx":
        missing = missing or find_spec("onnxruntime") is None or find_spec("optimum") is None
    if missing:
        return f'缺少当前策略的模型依赖，请执行 python -m pip install -e ".[{extra}]"，或切换为 BM25。'
    return None


def contains_api_key(value: str) -> bool:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return bool(key and key in value)


def evidence_rows(hits: list[SearchResult]) -> list[dict]:
    return [{"rank": rank, "chunk_id": hit.chunk.chunk_id, "source": hit.chunk.source,
             "text": hit.chunk.text, "score": hit.score, "start": hit.chunk.start,
             "end": hit.chunk.end} for rank, hit in enumerate(hits, 1)]


def _model_failure(exc: Exception) -> str:
    if isinstance(exc, TimeoutError) or (
        isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError)
    ):
        return "model_timeout"
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    return "model_error"


def run_fixed_rag(question: str, retriever: Retriever, config: dict,
                  generator: AnswerGenerator, trace_dir: Path) -> dict:
    request_id = uuid4().hex
    path = trace_dir / f"{request_id}.jsonl"
    trace = JsonlTrace(path)

    def record(event: str, **fields):
        trace.write({"event": event, "request_id": request_id,
                     "timestamp": datetime.now(timezone.utc).isoformat(), **fields})

    result = {"request_id": request_id, "trace_path": str(path), "status": "failed",
              "answer": None, "evidence": [], "citations": [], "model_calls": 0,
              "token_usage": None, "top_k": config["top_k"]}
    try:
        record("request_started", config=config, question=question)
        record("retrieval_started", top_k=config["top_k"])
        try:
            hits = retriever.search(question, config["top_k"])
        except Exception:
            result["stop_reason"] = "retrieval_failed"
            return result
        result["evidence"] = evidence_rows(hits)
        record("retrieval_finished", count=len(hits), evidence=result["evidence"])

        record("model_started", model_call=1)
        result["model_calls"] = 1
        model_started = perf_counter()
        try:
            generation = generator.generate(question, hits)
        except Exception as exc:
            result["stop_reason"] = _model_failure(exc)
            record("model_failed", model_call=1, error_type=type(exc).__name__,
                   elapsed_ms=(perf_counter() - model_started) * 1000)
            return result

        usage = asdict(generation.stats)
        latency = usage.pop("latency_seconds")
        # DeepSeekGenerator 用 0 表示缺失的计数。空用量留为未知。
        result["token_usage"] = usage if usage["total_tokens"] else None
        record("model_finished", model_call=1, elapsed_ms=latency * 1000, usage=result["token_usage"])
        answer = generation.text.strip()
        refs = sorted({int(ref) for ref in CITATION_PATTERN.findall(answer)})
        result.update(answer=answer, citations=[{"citation_id": i, **result["evidence"][i - 1]}
                                               for i in refs if 1 <= i <= len(hits)])
        if any(not 1 <= ref <= len(hits) for ref in refs):
            result["stop_reason"] = "invalid_citations"
        elif answer.startswith("根据当前资料无法确定"):
            result.update(status="abstained", stop_reason="model_abstained")
        elif not answer:
            result["stop_reason"] = "invalid_model_response"
        elif not refs:
            result["stop_reason"] = "missing_citations"
        else:
            result.update(status="completed", stop_reason="model_answered")
        return result
    finally:
        try:
            record("request_finished", **{key: value for key, value in result.items()
                                          if key not in {"request_id", "trace_path", "evidence"}})
        finally:
            trace.close()


def run_answer(mode: str, question: str, config: dict, documents: list[Document],
               retriever: Retriever, trace_dir: Path) -> dict:
    if mode == "rag":
        return run_fixed_rag(question, retriever, config, DeepSeekGenerator(), trace_dir)
    if mode == "langgraph":
        from domainrag.workflows.langgraph_agent import run_langgraph_agent
        runner = run_langgraph_agent
    else:
        runner = run_agent
    tools = KnowledgeTools(retriever, documents, config)
    return runner(question, tools, DeepSeekToolModel(), trace_dir)


def read_trace(path: Path) -> list[dict]:
    """只把界面使用的结构化事件和字段送到浏览器。"""
    fields = {
        "retrieval_started": ("top_k",), "retrieval_finished": ("count",),
        "model_started": ("model_call", "allow_tools"),
        "model_finished": ("model_call", "finish_reason", "elapsed_ms", "usage"),
        "model_failed": ("model_call", "error_type", "elapsed_ms"),
        "tool_started": ("tool", "call_id", "arguments"),
        "tool_finished": ("tool", "call_id", "elapsed_ms", "ok"),
        "tool_observation": ("tool", "provider_call_id", "observation"),
        "request_finished": ("tool_calls", "failed_calls"),
    }
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            event = row["event"]
            if event in fields:
                events.append({"event": event, **{key: row[key] for key in fields[event] if key in row}})
    return events
