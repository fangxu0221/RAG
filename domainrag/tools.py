"""只读知识工具与逐请求追踪；尚不包含模型自主调用循环。

运行：python -m domainrag.tools --help
"""

import argparse
import json
from pathlib import Path

from .application.knowledge_service import KnowledgeTools, ToolInputError, ToolSession
from .domain.protocols import TOOL_SCHEMAS

__all__ = ["KnowledgeTools", "ToolInputError", "ToolSession", "TOOL_SCHEMAS", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/tool_traces"))
    commands = parser.add_subparsers(dest="tool", required=True)
    search = commands.add_parser("search_knowledge")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=3)
    read = commands.add_parser("read_document")
    read.add_argument("--source", required=True)
    read.add_argument("--start", type=int, default=0)
    read.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()
    tools = KnowledgeTools.from_config(args.config)
    arguments = ({"query": args.query, "top_k": args.top_k} if args.tool == "search_knowledge"
                 else {"source": args.source, "start": args.start, "max_chars": args.max_chars})
    with ToolSession(tools, args.trace_dir) as session:
        output = session.call(args.tool, arguments)
    print(json.dumps({**output, "trace_path": str(session.path)}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if output["ok"] else 1)


if __name__ == "__main__":
    main()
