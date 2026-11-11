"""有界的DeepSeek工具调用循环；保留原固定RAG入口与冻结生成提示词。"""

import argparse
import json
from pathlib import Path

from .application.knowledge_service import KnowledgeTools, ToolSession
from .domain.protocols import ChatModel, TOOL_SCHEMAS
from .infrastructure.llm.tool_model import DeepSeekToolModel
from .workflows.native_agent import AGENT_PROMPT, AgentLimits, run_agent

__all__ = [
    "AGENT_PROMPT", "AgentLimits", "ChatModel", "DeepSeekToolModel",
    "KnowledgeTools", "TOOL_SCHEMAS", "ToolSession", "run_agent", "main",
]


def _argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--trace-dir", type=Path, default=Path("results/agent_traces"))
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=90)
    return parser


def _execute_cli(args, runner) -> None:
    limits = AgentLimits(args.max_tool_calls, args.max_model_calls, args.max_seconds)
    model = DeepSeekToolModel(args.model)
    tools = KnowledgeTools.from_config(args.config)
    result = runner(args.question, tools, model, args.trace_dir, limits)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] in {"completed", "abstained"} else 1)


def main() -> None:
    _execute_cli(_argument_parser(__doc__).parse_args(), run_agent)


if __name__ == "__main__":
    main()
