"""Embedding 稠密检索：把问题和 Chunk 映射到同一个向量空间后比较相似度。

初始化时一次性编码全部 Chunk；每次查询只编码问题，然后用归一化向量的点积
（等价于余弦相似度）排序。当前向量只保存在内存，未来可由向量数据库替换。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from ...models import Chunk, SearchResult


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Encoder(Protocol):
    """SentenceTransformer 及测试用假 Encoder 共同遵守的接口。"""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


def _as_vector(value: object) -> list[float]:
    """把 numpy 数组或普通序列统一转换为 Python 浮点列表。"""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("encoder output must be a sequence of vectors")
    return [float(item) for item in value]


def _normalize(vector: list[float]) -> list[float]:
    """进行 L2 归一化，使向量长度为 1。"""

    # ||v||₂ = sqrt(v1² + v2² + ... + vn²)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingRetriever:
    """使用本地 Sentence Transformers 模型执行稠密语义检索。"""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 32,
        encoder: Encoder | None = None,
    ):
        if not chunks:
            raise ValueError("EmbeddingRetriever requires at least one chunk")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if encoder is None:
            # 正常运行时加载真实模型；测试时可传入轻量 FakeEncoder，避免下载模型。
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Embedding retrieval requires sentence-transformers. "
                    'Install it with: python -m pip install -e ".[embedding]"'
                ) from exc
            encoder = SentenceTransformer(model_name)

        self.chunks = chunks
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.encoder = encoder
        # 文档向量只在初始化时计算一次，后续多个问题可以复用。
        self.chunk_embeddings = self._encode([chunk.text for chunk in chunks])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本，并防御模型返回类型或数量异常。"""

        encoded = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if not isinstance(encoded, Sequence):
            raise TypeError("encoder output must contain one vector per input text")
        # 即使底层模型声称已经 normalize，也再次规范化，确保自定义 Encoder 行为一致。
        vectors = [_normalize(_as_vector(vector)) for vector in encoded]
        if len(vectors) != len(texts):
            raise ValueError("encoder returned a different number of vectors than inputs")
        return vectors

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """编码问题、计算它与所有 Chunk 的余弦相似度，并取 Top-K。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # BGE 检索模型建议只给“问题”添加检索指令，文档 Chunk 不添加。
        query_text = f"{self.query_instruction}{query}" if self.query_instruction else query
        query_vector = self._encode([query_text])[0]
        scored: list[SearchResult] = []
        for chunk, chunk_vector in zip(self.chunks, self.chunk_embeddings):
            if len(chunk_vector) != len(query_vector):
                raise ValueError("query and document embedding dimensions do not match")
            # 两个向量都已归一化，因此点积就是 cosine similarity；越大越相关。
            score = sum(
                query_value * chunk_value
                for query_value, chunk_value in zip(query_vector, chunk_vector)
            )
            scored.append(SearchResult(chunk=chunk, score=score))

        # 相同分数使用 chunk_id 打破平局，保证每次运行顺序一致。
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[: min(top_k, len(scored))]
