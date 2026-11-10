"""知识快照服务与逐请求工具会话。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ..domain.protocols import Retriever, TOOL_SCHEMAS
from ..infrastructure.storage.trace import JsonlTrace
from ..models import Document
from ..text import normalize_text


class ToolInputError(ValueError):
    """可返回给调用者修正的工具参数错误。"""


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    # bool是int的子类，这里必须显式拒绝True/False。
    if type(value) is not int or not minimum <= value <= maximum:
        raise ToolInputError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


class KnowledgeTools:
    """长期复用的检索器和只读文档快照；每个请求另建ToolSession。"""

    def __init__(self, retriever: Retriever, documents: list[Document], config: dict):
        self._retriever = retriever
        self._documents = {doc.source: normalize_text(doc.text) for doc in documents}
        if len(self._documents) != len(documents):
            raise ValueError("duplicate document source IDs")
        self.config = deepcopy(config)
        payload = json.dumps(
            sorted((doc.source, doc.text) for doc in documents), ensure_ascii=False
        )
        self.corpus_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_config(cls, config_path: Path) -> KnowledgeTools:
        # 兼容从配置创建服务的旧入口；常规构造函数接收已装配依赖。
        from ..bootstrap import build_retriever_from_documents, load_retrieval_snapshot

        config, documents = load_retrieval_snapshot(config_path)
        retriever = build_retriever_from_documents(documents, config)
        return cls(retriever, documents, config)

    def search_knowledge(self, query: str, top_k: int = 3) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ToolInputError("query must contain 1 to 2000 characters")
        top_k = _integer(top_k, "top_k", 1, 10)
        results = []
        for rank, hit in enumerate(self._retriever.search(query.strip(), top_k), 1):
            chunk = hit.chunk
            if chunk.source not in self._documents or not math.isfinite(hit.score):
                raise RuntimeError("retriever returned invalid evidence")
            # 既有切块会strip，start/end仍指向规范化全文中的窗口。
            if self._documents[chunk.source][chunk.start:chunk.end].strip() != chunk.text:
                raise RuntimeError("retriever evidence differs from document snapshot")
            results.append({
                "rank": rank, "chunk_id": chunk.chunk_id, "source": chunk.source,
                "text": chunk.text[:2000], "text_truncated": len(chunk.text) > 2000,
                "score": hit.score, "start": chunk.start, "end": chunk.end,
            })
        return {"results": results, "count": len(results),
                "offset_basis": "normalized_text", "score_is_probability": False}

    def read_document(self, source: str, start: int = 0, max_chars: int = 2000) -> dict:
        # 不把用户输入拼接成文件路径；只查启动时加载的来源映射。
        if not isinstance(source, str) or source not in self._documents:
            raise ToolInputError("source must be an exact known document ID")
        text = self._documents[source]
        start = _integer(start, "start", 0, len(text))
        max_chars = _integer(max_chars, "max_chars", 1, 4000)
        end = min(start + max_chars, len(text))
        return {"source": source, "text": text[start:end], "start": start, "end": end,
                "total_chars": len(text), "next_start": end if end < len(text) else None,
                "offset_basis": "normalized_text"}


class ToolSession:
    """单请求的工具执行与落盘；调用方通过call()获得成功/失败统一结构。"""

    def __init__(self, tools: KnowledgeTools, trace_dir: Path):
        self.tools = tools
        self.request_id = uuid4().hex
        self.path = trace_dir / f"{self.request_id}.jsonl"
        self._trace = JsonlTrace(self.path)
        self._started = perf_counter()
        self._calls = 0
        self._errors = 0
        self._closed = False
        self._outcome = {}
        self._write({"event": "request_started", "schema_version": 1,
                     "config": deepcopy(tools.config), "corpus_sha256": tools.corpus_sha256,
                     "generation_model": None, "token_usage": None})

    def _write(self, event: dict) -> None:
        event = {"request_id": self.request_id,
                 "timestamp": datetime.now(timezone.utc).isoformat(), **event}
        self._trace.write(event)

    def record(self, event: dict) -> None:
        """给Agent层复用同一个request_id记录模型事件，不记录凭据。"""
        if self._closed:
            raise RuntimeError("session is closed")
        self._write(event)

    def set_outcome(self, outcome: dict) -> None:
        """设置Agent终态；工具独立调用时仍保留原来的默认终态。"""
        keys = {"status", "stop_reason", "answer", "citations", "token_usage", "model_calls"}
        self._outcome = {key: deepcopy(value) for key, value in outcome.items() if key in keys}

    def call(self, name: str, arguments: dict) -> dict:
        if self._closed:
            raise RuntimeError("session is closed")
        # 边界只接收JSON对象；先独立快照，防止调用者之后修改参数污染日志。
        if not isinstance(arguments, dict):
            raise ToolInputError("arguments must be a JSON object")
        arguments = json.loads(json.dumps(arguments, allow_nan=False))
        self._calls += 1
        call_id = f"{self.request_id}:{self._calls}"
        self._write({"event": "tool_started", "call_id": call_id,
                     "tool": name, "arguments": arguments})
        started = perf_counter()
        try:
            schema = next((s["function"] for s in TOOL_SCHEMAS
                           if s["function"]["name"] == name), None)
            if schema is None:
                raise ToolInputError("unknown tool")
            params = schema["parameters"]
            if set(arguments) - set(params["properties"]):
                raise ToolInputError("unexpected argument")
            if set(params["required"]) - set(arguments):
                raise ToolInputError("missing required argument")
            result = getattr(self.tools, name)(**arguments)
            output = {"ok": True, "result": result, "error": None}
        except Exception as exc:
            self._errors += 1
            # 对未知后端异常不暴露可能包含路径、密钥的原始消息。
            output = {"ok": False, "result": None, "error": {
                "code": "invalid_arguments" if isinstance(exc, ToolInputError) else "tool_failed",
                "type": type(exc).__name__,
                "message": str(exc) if isinstance(exc, ToolInputError) else "tool execution failed",
            }}
        output.update(request_id=self.request_id, call_id=call_id)
        self._write({"event": "tool_finished", "call_id": call_id, "tool": name,
                     "elapsed_ms": (perf_counter() - started) * 1000, **output})
        return output

    def close(self, aborted: bool = False) -> None:
        if self._closed:
            return
        try:
            self._write({"event": "request_finished", "tool_calls": self._calls,
                         "failed_calls": self._errors,
                         "status": "aborted" if aborted else "completed_with_errors" if self._errors else "completed",
                         "stop_reason": "caller_aborted" if aborted else "caller_finished",
                         "elapsed_ms": (perf_counter() - self._started) * 1000,
                         "answer": None, "citations": [], "token_usage": None,
                         **({} if aborted else self._outcome)})
        finally:
            self._closed = True
            self._trace.close()

    def __enter__(self) -> ToolSession:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(aborted=exc_type is not None)
