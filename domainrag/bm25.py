"""BM25 稀疏检索：根据问题词与 Chunk 词语的重合程度进行排序。

它擅长精确关键词、专有名词和编号匹配。与 Embedding 的“语义相似”不同，BM25
主要使用词频、词的稀有程度和文档长度三个信号。
"""

from __future__ import annotations

import math
from collections import Counter

from .models import Chunk, SearchResult
from .text import tokenize


class BM25Retriever:
    """预先统计语料词频，并在查询时计算每个 Chunk 的 BM25 分数。"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # TF：每个 Chunk 内部各 token 出现了多少次。
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        # 平均长度用于惩罚“只是因为篇幅长而碰巧包含更多关键词”的 Chunk。
        self.average_length = sum(self.lengths) / len(self.lengths)

        # DF：某 token 出现在多少个不同 Chunk 中；同一 Chunk 中重复多次也只算一次。
        self.document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequencies.update(frequencies.keys())

    def _idf(self, term: str) -> float:
        """计算逆文档频率：越少见的词，区分文档的能力越强，权重越高。"""

        document_frequency = self.document_frequencies.get(term, 0)
        corpus_size = len(self.chunks)
        return math.log(
            1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def score(self, query: str, document_index: int) -> float:
        """计算一个查询与指定 Chunk 的 BM25 总分。"""

        query_terms = tokenize(query)
        frequencies = self.term_frequencies[document_index]
        document_length = self.lengths[document_index]
        score = 0.0

        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            # b 控制长度归一化强度；k1 控制词频增加时分数多快趋于饱和。
            normalization = self.k1 * (
                1 - self.b + self.b * document_length / max(self.average_length, 1e-12)
            )
            # 每个命中词贡献：稀有程度(IDF) × 经过饱和与长度修正的词频(TF)。
            score += self._idf(term) * frequency * (self.k1 + 1) / (frequency + normalization)
        return score

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """遍历全部 Chunk 计算 BM25，并返回得分最高的 top_k 个。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [
            SearchResult(chunk=chunk, score=self.score(query, index))
            for index, chunk in enumerate(self.chunks)
        ]
        # 分数降序；分数相同时按 chunk_id 排序，使结果稳定、便于回归测试。
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[: min(top_k, len(scored))]
