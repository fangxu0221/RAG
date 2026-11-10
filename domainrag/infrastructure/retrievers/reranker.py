"""Cross-Encoder 重排器：对第一阶段召回的少量候选进行更精细的相关性判断。

BM25/Embedding 适合从大量 Chunk 中快速“召回”；Cross-Encoder 会把 query 和每个
候选 Chunk 一起送入模型，判断更准但计算更慢，所以只处理 candidate_k 个候选。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ...models import SearchResult
from ...domain.protocols import Retriever


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANKER_BACKEND = "onnx"
DEFAULT_RERANKER_FILE = "onnx/model_quint8_avx2.onnx"


class PairScorer(Protocol):
    """真实 CrossEncoder 与测试用假打分器共同遵守的接口。"""

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


def _as_scores(value: object) -> list[float]:
    """兼容 numpy、[score] 等模型输出形状，最终统一成一维浮点分数。"""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("reranker output must be a sequence of scores")

    scores: list[float] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 1:
                raise ValueError("reranker must return one score per pair")
            item = item[0]
        scores.append(float(item))
    return scores


class CrossEncoderReranker:
    """联合编码 query-chunk 对，对第一阶段候选重新排序。

    这个类也实现 Retriever.search()，因此对 CLI、生成器和评测器来说，它仍然只是
    一个普通检索器；装饰器式设计避免后续代码关心是否开启了重排。
    """

    def __init__(
        self,
        retriever: Retriever,
        model_name: str = DEFAULT_RERANKER_MODEL,
        candidate_k: int = 10,
        batch_size: int = 4,
        max_length: int = 512,
        backend: str = DEFAULT_RERANKER_BACKEND,
        model_file: str | None = None,
        scorer: PairScorer | None = None,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if backend not in {"torch", "onnx", "openvino"}:
            raise ValueError("backend must be 'torch', 'onnx', or 'openvino'")

        # ONNX 默认选择量化模型文件；Torch/OpenVINO 不应误用 ONNX 文件路径。
        selected_model_file = (
            model_file
            if model_file is not None or backend != "onnx"
            else DEFAULT_RERANKER_FILE
        )
        if scorer is None:
            # 测试可以注入 FakeScorer；真实运行才加载可能较重的模型依赖。
            try:
                from sentence_transformers import CrossEncoder
                model_kwargs = (
                    {"file_name": selected_model_file}
                    if selected_model_file
                    else None
                )
                scorer = CrossEncoder(
                    model_name,
                    max_length=max_length,
                    backend=backend,
                    model_kwargs=model_kwargs,
                )
            except ImportError as exc:
                extra = "reranker" if backend in {"onnx", "openvino"} else "embedding"
                raise RuntimeError(
                    f"Cross-Encoder backend {backend!r} is not installed. "
                    f'Install it with: python -m pip install -e ".[{extra}]"'
                ) from exc

        self.retriever = retriever
        self.model_name = model_name
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.backend = backend
        self.model_file = selected_model_file
        self.scorer = scorer

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """先召回候选，再以 Cross-Encoder 新分数重排并截取 Top-K。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # 第一阶段召回量至少要覆盖最终需要返回的 top_k。
        requested_candidates = max(top_k, self.candidate_k)
        candidates = self.retriever.search(query, top_k=requested_candidates)
        if not candidates:
            return []

        # Cross-Encoder 同时阅读问题与文档，因此可以建模更细粒度的词间交互。
        pairs = [(query, result.chunk.text) for result in candidates]
        scores = _as_scores(
            self.scorer.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )
        if len(scores) != len(candidates):
            raise ValueError("reranker returned a different number of scores than pairs")

        # 保留原 Chunk，但用第二阶段分数覆盖第一阶段分数。
        reranked = [
            SearchResult(chunk=result.chunk, score=score)
            for result, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return reranked[: min(top_k, len(reranked))]
