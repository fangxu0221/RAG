"""所有检索器共同遵守的最小接口。"""

from __future__ import annotations

from typing import Protocol

from .models import SearchResult


class Retriever(Protocol):
    """检索器协议：输入问题和 top_k，输出按相关性排序的 SearchResult。

    Protocol 是结构化类型约束：类不必显式继承 Retriever，只要实现同样的 search()
    就能被 Hybrid、Reranker 和评测代码使用。这也是以后接向量数据库的扩展点。
    """

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...
