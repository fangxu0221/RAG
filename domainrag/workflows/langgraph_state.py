"""LangGraph V1 的请求状态与原始模型消息。"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """节点返回部分更新；每次请求初始化独立的容器。"""

    question: str
    messages: Annotated[list[dict], operator.add]
    evidence: list[dict]
    evidence_keys: dict[str, int]
    calls_seen: list[str]
    ids_seen: list[str]
    model_calls: int
    tool_attempts: int
    known_token_usage: dict[str, int]
    usage_complete: bool
    pending_message: dict | None
    finish_reason: str | None
    status: str | None
    stop_reason: str | None
    answer: str | None
    citations: list[dict]
    result: dict
