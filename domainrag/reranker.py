"""旧 reranker 导入路径。"""

from .infrastructure.retrievers.reranker import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_BACKEND,
    DEFAULT_RERANKER_FILE,
    PairScorer,
    CrossEncoderReranker,
)

__all__ = ['DEFAULT_RERANKER_MODEL', 'DEFAULT_RERANKER_BACKEND', 'DEFAULT_RERANKER_FILE', 'PairScorer', 'CrossEncoderReranker']
