"""使用 StateGraph 编排逐请求的工具调用。"""

from pathlib import Path
from time import perf_counter
from typing import Literal

from ..application.knowledge_service import KnowledgeTools, ToolSession
from ..domain.protocols import ChatModel
from .langgraph_state import AgentState
from .native_agent import (
    AgentLimits, _answer_outcome, _call_model, _finish, _initial_messages,
    _json, _record_agent_start, _tool_batch_error, _tool_message,
)


def _route_after_model(state: AgentState) -> Literal["tools", "validate_answer", "finalize"]:
    if state["status"] is not None:
        return "finalize"
    message = state["pending_message"]
    if message is not None and message.get("tool_calls"):
        return "tools"
    return "validate_answer"


def _route_after_tools(state: AgentState) -> Literal["model", "finalize"]:
    return "finalize" if state["status"] is not None else "model"


def _build_graph(session: ToolSession, model: ChatModel, limits: AgentLimits, started: float):
    from langgraph.graph import END, START, StateGraph

    def model_node(state: AgentState) -> AgentState:
        if state["model_calls"] >= limits.max_model_calls:
            return {"stop_reason": "model_budget", "status": "stopped"}
        remaining = limits.max_seconds - (perf_counter() - started)
        if remaining <= 0:
            return {"stop_reason": "time_budget", "status": "stopped"}
        if len(_json(state["messages"])) > limits.max_context_chars:
            return {"stop_reason": "context_budget", "status": "stopped"}

        model_calls = state["model_calls"] + 1
        remaining_attempts = limits.max_tool_calls - state["tool_attempts"]
        message, reason, usage, usage_complete = _call_model(
            session, model, state["messages"], model_calls, remaining_attempts > 0, remaining,
            state["known_token_usage"], state["usage_complete"],
        )
        update: AgentState = {
            "model_calls": model_calls, "pending_message": message, "finish_reason": reason,
            "known_token_usage": usage, "usage_complete": usage_complete,
        }
        if message is None:
            update.update(stop_reason="model_error", status="failed")
        elif perf_counter() - started >= limits.max_seconds:
            update.update(stop_reason="time_budget", status="stopped")
        elif calls := message.get("tool_calls"):
            update.update(_tool_batch_error(calls, reason, remaining_attempts, state["ids_seen"]))
        return update

    def tools_node(state: AgentState) -> AgentState:
        message = state["pending_message"]
        assert message is not None
        new_messages = [message]
        evidence = state["evidence"].copy()
        evidence_keys = state["evidence_keys"].copy()
        calls_seen = set(state["calls_seen"])
        ids_seen = state["ids_seen"].copy()
        attempted = state["tool_attempts"]
        update: AgentState = {"pending_message": None}
        for call in message["tool_calls"]:
            if perf_counter() - started >= limits.max_seconds:
                update.update(stop_reason="time_budget", status="stopped")
                break
            attempted += 1
            ids_seen.append(call["id"])
            new_messages.append(_tool_message(call, session, calls_seen, evidence, evidence_keys))
        update.update(messages=new_messages, evidence=evidence, evidence_keys=evidence_keys,
                      calls_seen=sorted(calls_seen), ids_seen=ids_seen, tool_attempts=attempted)
        return update

    def validate_answer_node(state: AgentState) -> AgentState:
        message = state["pending_message"]
        assert message is not None
        return _answer_outcome(message, state["finish_reason"], state["evidence"])

    def finalize_node(state: AgentState) -> AgentState:
        return {"result": _finish(
            session, state["evidence"], state["model_calls"], state["tool_attempts"],
            state["known_token_usage"], state["usage_complete"],
            stop_reason=state["stop_reason"], status=state["status"],
            answer=state["answer"], citations=state["citations"],
        )}

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("validate_answer", validate_answer_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", _route_after_model)
    graph.add_conditional_edges("tools", _route_after_tools)
    graph.add_edge("validate_answer", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_langgraph_agent(question: str, tools: KnowledgeTools, model: ChatModel,
                        trace_dir: Path, limits: AgentLimits | None = None, *,
                        answer_style: str | None = None) -> dict:
    """以与原生 run_agent 相同的契约执行一次 LangGraph 请求。"""
    limits = limits or AgentLimits()
    messages = _initial_messages(question, answer_style=answer_style)
    # LangGraph 间接安装此客户端，显式关闭环境变量可开启的远程追踪。
    from langsmith import tracing_context

    state: AgentState = {
        "question": question, "messages": messages,
        "evidence": [], "evidence_keys": {}, "calls_seen": [], "ids_seen": [],
        "model_calls": 0, "tool_attempts": 0,
        "known_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "usage_complete": True, "pending_message": None, "finish_reason": None,
        "status": None, "stop_reason": None, "answer": None, "citations": [],
    }
    with tracing_context(enabled=False):
        started = perf_counter()
        with ToolSession(tools, trace_dir) as session:
            _record_agent_start(session, model, question, limits, messages[0]["content"])
            graph = _build_graph(session, model, limits, started)
            # 每次模型调用至多对应一个工具节点，另给预算检查、验证和收尾留余量。
            result = graph.invoke(state, config={"recursion_limit": 2 * limits.max_model_calls + 4})
            return result["result"]
