"""贯穿整个 RAG 流程的核心数据结构。

阅读其他模块时，可以先记住数据的变化顺序：
Document（整篇文档） -> Chunk（切分后的文本块） -> SearchResult（带分数的检索结果）。
EvaluationCase 则保存一道评测题及其标准标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """一篇尚未切分的原始文档。

    source 是相对于文档目录的路径，也是后续评测使用的“文档身份”；
    text 是文件中读取到的完整文本。
    """

    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """从 Document 中截取出的一个文本块。

    chunk_id 必须稳定且唯一，source 用来追溯原文档；start/end 记录该块在
    规范化后全文中的字符区间，便于定位原文。检索真正排序的是 Chunk，而不是整篇文档。
    """

    chunk_id: str
    source: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SearchResult:
    """一个检索命中：既保留 Chunk，也携带当前检索器给出的相关性分数。"""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class EvaluationCase:
    """一条离线评测样本。

    relevant_sources 是应该被检索到的正确文档集合；expected_answer 只在答案评测
    中使用；metadata 可保存 question_type 等不固定的扩展字段。
    """

    case_id: str
    question: str
    relevant_sources: tuple[str, ...]
    expected_answer: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    # False 表示整个冻结知识库缺少答案；不能仅因 Top-K 漏召回就改成 False。
    answerable: bool = True
