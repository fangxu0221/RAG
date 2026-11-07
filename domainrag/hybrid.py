"""Hybrid RRF 检索：融合 BM25 与 Embedding 等多个检索器的排名。

不同检索器的原始分数量纲不可直接相加，例如 BM25 可能是 5.2，而余弦相似度
可能是 0.78。RRF 只使用“名次”而非原始分数，因此能稳定地融合异构检索器。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import Chunk, SearchResult
from .retriever import Retriever


class HybridRetriever:
    """使用 Reciprocal Rank Fusion（倒数排名融合）合并多路排序。"""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever requires at least two retrievers")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

        self.retrievers = tuple(retrievers)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """从每个子检索器取候选集，累加 RRF 分数后返回最终 Top-K。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # 候选数绝不能比最终 top_k 小，否则无法返回足够结果。
        requested_candidates = max(top_k, self.candidate_k)
        fused_scores: defaultdict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}

        for retriever in self.retrievers:
            # 防御某个子检索器重复返回同一 Chunk，避免单路结果被重复加分。
            seen_chunk_ids: set[str] = set()
            results = retriever.search(query, top_k=requested_candidates)
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunks.setdefault(chunk_id, result.chunk)
                # RRF 公式：score(d) += 1 / (rrf_k + rank)。
                # 同一 Chunk 在多路检索中都排名靠前时，会累计得到更高总分。
                fused_scores[chunk_id] += 1.0 / (self.rrf_k + rank)

        fused = [
            SearchResult(chunk=chunks[chunk_id], score=score)
            for chunk_id, score in fused_scores.items()
        ]
        # 分数降序，chunk_id 作为稳定的平局规则。
        fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return fused[: min(top_k, len(fused))]
