"""通过 LangGraph 运行 DomainRAG 工具调用 Agent。"""

from importlib.util import find_spec

from .agent import _argument_parser, _execute_cli
from .workflows.langgraph_agent import run_langgraph_agent


def main() -> None:
    parser = _argument_parser(__doc__)
    args = parser.parse_args()
    if find_spec("langgraph") is None:
        parser.error('LangGraph is required. Install with: python -m pip install -e ".[langgraph]"')
    _execute_cli(args, run_langgraph_agent)


if __name__ == "__main__":
    main()
