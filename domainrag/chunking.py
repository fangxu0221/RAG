"""把整篇文档切成可检索的重叠 Chunk。

当前实现按“字符数”切分，是一个容易理解和复现的基线。重叠区域用于减少关键信息
恰好被切分边界截断的问题。
"""

from __future__ import annotations

from .models import Chunk, Document
from .text import normalize_text


def chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """将单篇文档按固定字符窗口切分，并生成稳定的顺序编号。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    text = normalize_text(document.text)
    if not text:
        return []

    chunks: list[Chunk] = []
    # 例：chunk_size=420、overlap=80，则下一块向前移动 340 个字符。
    step = chunk_size - chunk_overlap
    start = 0
    index = 0
    while start < len(text):
        # 最后一块不足 chunk_size 时，end 停在全文末尾而不是越界。
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    # source + 顺序号让每个块可被引用，例如 alarm.md#0002。
                    chunk_id=f"{document.source}#{index:04d}",
                    source=document.source,
                    text=chunk_text,
                    start=start,
                    end=end,
                )
            )
            index += 1
        # 已经覆盖到全文末尾就立即结束，避免 overlap 导致重复的尾块。
        if end == len(text):
            break
        start += step
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """批量切分多篇文档，并合并为统一 Chunk 列表。"""

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks
