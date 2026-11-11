"""配置、冻结校验和检索器装配。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .infrastructure.retrievers.bm25 import BM25Retriever
from .chunking import chunk_documents
from .infrastructure.retrievers.embedding import DEFAULT_MODEL, DEFAULT_QUERY_INSTRUCTION, EmbeddingRetriever
from .infrastructure.retrievers.hybrid import HybridRetriever
from .io import load_config, load_documents
from .models import Chunk, Document
from .infrastructure.retrievers.reranker import (
    DEFAULT_RERANKER_BACKEND,
    DEFAULT_RERANKER_FILE,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from .domain.protocols import Retriever
from .frozen import verify_frozen_dataset


def project_path(config_path: Path, value: str) -> Path:
    """把配置中的项目相对路径解析为不依赖当前终端位置的绝对路径。"""

    # 配置放在 config/ 目录中，因此其父目录的父目录就是项目根目录。
    project_root = config_path.resolve().parent.parent
    return (project_root / value).resolve()


def build_bm25(chunks: list[Chunk], config: dict[str, object]) -> BM25Retriever:
    """从配置读取 BM25 超参数并构造稀疏检索器。"""

    bm25_config = dict(config.get("bm25", {}))
    return BM25Retriever(
        chunks,
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )


def build_embedding(
    chunks: list[Chunk], config: dict[str, object], *, prefer_cached_model: bool = False
) -> EmbeddingRetriever:
    """从配置读取模型、查询指令和批大小并构造稠密检索器。"""

    embedding_config = dict(config.get("embedding", {}))
    return EmbeddingRetriever(
        chunks,
        model_name=str(embedding_config.get("model", DEFAULT_MODEL)),
        query_instruction=str(
            embedding_config.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)
        ),
        batch_size=int(embedding_config.get("batch_size", 32)),
        prefer_cached_model=prefer_cached_model,
    )


def load_retrieval_snapshot(config_path: Path) -> tuple[dict[str, object], list[Document]]:
    """先校验冻结清单，再读取供检索、工具和评测共同使用的文档快照。"""
    config = load_config(config_path)
    verify_frozen_dataset(config_path)
    documents = load_documents(project_path(config_path, str(config["docs_path"])))
    return config, documents


def build_retriever(config_path: Path) -> tuple[Retriever, dict[str, object]]:
    """从配置装配检索器，并保留原配置供报告引用。"""
    config, documents = load_retrieval_snapshot(config_path)
    return build_retriever_from_documents(documents, config), config


def build_retriever_from_documents(
    documents: list[Document], config: dict[str, object], *,
    embedding_builder: Callable[[list[Chunk], dict[str, object]], Retriever] | None = None,
) -> Retriever:
    """从调用方已加载的快照装配；文件加载和冻结检查由入口负责。"""

    # 2. 按相同 chunk_size/overlap 切成所有检索器共享的 Chunk。
    chunks = chunk_documents(
        documents,
        chunk_size=int(config["chunk_size"]),
        chunk_overlap=int(config["chunk_overlap"]),
    )
    # 3. 根据配置选择第一阶段召回器。
    build_dense = embedding_builder or build_embedding
    retriever_name = str(config.get("retriever", "bm25")).lower()
    if retriever_name == "bm25":
        retriever: Retriever = build_bm25(chunks, config)
    elif retriever_name == "embedding":
        retriever = build_dense(chunks, config)
    elif retriever_name == "hybrid":
        hybrid_config = dict(config.get("hybrid", {}))
        # Hybrid 同时执行 BM25 与 Embedding，再使用 RRF 融合名次。
        retriever = HybridRetriever(
            [build_bm25(chunks, config), build_dense(chunks, config)],
            candidate_k=int(hybrid_config.get("candidate_k", 10)),
            rrf_k=int(hybrid_config.get("rrf_k", 60)),
        )
    else:
        raise ValueError(
            f"unsupported retriever {retriever_name!r}; "
            "choose 'bm25', 'embedding', or 'hybrid'"
        )

    # 4. 若启用 Reranker，就把第一阶段检索器包在 CrossEncoderReranker 内部。
    reranker_config = dict(config.get("reranker", {}))
    if bool(reranker_config.get("enabled", False)):
        reranker_backend = str(
            reranker_config.get("backend", DEFAULT_RERANKER_BACKEND)
        )
        configured_model_file = reranker_config.get("model_file")
        # ONNX 使用配置文件或默认量化模型；其他后端不需要 ONNX 文件。
        reranker_model_file = (
            str(configured_model_file)
            if configured_model_file
            else DEFAULT_RERANKER_FILE if reranker_backend == "onnx" else None
        )
        retriever = CrossEncoderReranker(
            retriever,
            model_name=str(
                reranker_config.get("model", DEFAULT_RERANKER_MODEL)
            ),
            candidate_k=int(reranker_config.get("candidate_k", 10)),
            batch_size=int(reranker_config.get("batch_size", 4)),
            max_length=int(reranker_config.get("max_length", 512)),
            backend=reranker_backend,
            model_file=reranker_model_file,
        )
    return retriever
