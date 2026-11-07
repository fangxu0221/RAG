"""文件输入输出层：负责读取知识库、评测集和配置，并写出 JSON 报告。

这个模块不做切分、检索或评测，只负责把磁盘数据转换成 models.py 中的数据对象。
把 I/O 与算法分开后，后续增加 PDF 解析或数据库存储时不会污染检索逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Document, EvaluationCase


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(root: str | Path) -> list[Document]:
    """递归读取目录下所有 Markdown/TXT 文件，返回完整文档列表。"""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"document path does not exist: {root_path}")

    documents: list[Document] = []
    # 排序保证不同机器上的读取顺序一致，让 chunk_id 和评测结果可复现。
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        # 只保存相对路径，避免报告中出现某台电脑独有的绝对路径。
        source = path.relative_to(root_path).as_posix()
        documents.append(Document(source=source, text=text))
    if not documents:
        raise ValueError(f"no Markdown or TXT documents found under {root_path}")
    return documents


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """读取 JSONL 评测集；每一行是一个独立 JSON 对象。"""

    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        # 记录行号，数据字段缺失时可以直接定位到原评测文件。
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            answerable = item.get("answerable", True)
            if not isinstance(answerable, bool):
                raise ValueError(f"answerable must be a boolean at line {line_number}")
            sources = item.get("relevant_sources")
            if not isinstance(sources, list) or any(not isinstance(s, str) or not s for s in sources):
                raise ValueError(f"relevant_sources must be a list of source names at line {line_number}")
            if answerable != bool(sources):
                raise ValueError(f"answerable and relevant_sources disagree at line {line_number}")
            try:
                cases.append(
                    EvaluationCase(
                        case_id=str(item["id"]),
                        question=str(item["question"]),
                        relevant_sources=tuple(item["relevant_sources"]),
                        expected_answer=item.get("expected_answer"),
                        metadata=item.get("metadata", {}),
                        answerable=answerable,
                    )
                )
            except KeyError as exc:
                raise ValueError(f"missing {exc.args[0]!r} at line {line_number}") from exc
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    return cases


def load_config(path: str | Path) -> dict[str, object]:
    """将 JSON 配置读取为字典，具体参数的解释和默认值由 cli.py 负责。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    """以便于人工阅读的 UTF-8 JSON 格式保存评测报告。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
