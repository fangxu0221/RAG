"""旧 embedding 导入路径。"""

from .infrastructure.retrievers.embedding import (
    DEFAULT_MODEL,
    DEFAULT_QUERY_INSTRUCTION,
    Encoder,
    EmbeddingRetriever,
)

__all__ = ['DEFAULT_MODEL', 'DEFAULT_QUERY_INSTRUCTION', 'Encoder', 'EmbeddingRetriever']
