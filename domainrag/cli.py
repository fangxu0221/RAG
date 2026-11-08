"""命令行入口和对象装配层。

这个文件不实现检索公式，而是根据 JSON 配置把“加载文档 -> 切分 -> 选择检索器
-> 可选重排 -> 查询/生成/评测”串起来。复习项目总流程时可以从 main() 向上阅读。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer_evaluation import DeepSeekAnswerJudge, evaluate_answers
from .bm25 import BM25Retriever
from .chunking import chunk_documents
from .embedding import DEFAULT_MODEL, DEFAULT_QUERY_INSTRUCTION, EmbeddingRetriever
from .evaluation import evaluate_retrieval
from .generator import DeepSeekGenerator, OllamaGenerator
from .hybrid import HybridRetriever
from .io import load_config, load_documents, load_evaluation_cases, write_json
from .models import Chunk, Document
from .reranker import (
    DEFAULT_RERANKER_BACKEND,
    DEFAULT_RERANKER_FILE,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from .retriever import Retriever
from .frozen import verify_frozen_dataset, runtime_fingerprint, text_sha256


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
    chunks: list[Chunk], config: dict[str, object]
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
    )


def build_retriever(config_path: Path) -> tuple[Retriever, dict[str, object]]:
    """构建完整检索流水线，是整个项目最重要的“装配函数”。

    返回值同时包含最终检索器和原配置，后续报告会把配置原样写入，保证实验可追溯。
    """

    config = load_config(config_path)
    # 带冻结清单的配置在编码/模型调用前检查，普通开发集不受影响。
    verify_frozen_dataset(config_path)
    # 1. 读取知识库整篇文档。
    documents = load_documents(project_path(config_path, str(config["docs_path"])))
    return build_retriever_from_documents(documents, config), config


def build_retriever_from_documents(
    documents: list[Document], config: dict[str, object]
) -> Retriever:
    """从调用方已加载的快照装配；文件加载和冻结检查由入口负责。"""

    # 2. 按相同 chunk_size/overlap 切成所有检索器共享的 Chunk。
    chunks = chunk_documents(
        documents,
        chunk_size=int(config["chunk_size"]),
        chunk_overlap=int(config["chunk_overlap"]),
    )
    # 3. 根据配置选择第一阶段召回器。
    retriever_name = str(config.get("retriever", "bm25")).lower()
    if retriever_name == "bm25":
        retriever: Retriever = build_bm25(chunks, config)
    elif retriever_name == "embedding":
        retriever = build_embedding(chunks, config)
    elif retriever_name == "hybrid":
        hybrid_config = dict(config.get("hybrid", {}))
        # Hybrid 同时执行 BM25 与 Embedding，再使用 RRF 融合名次。
        retriever = HybridRetriever(
            [build_bm25(chunks, config), build_embedding(chunks, config)],
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


def run_benchmark(args: argparse.Namespace) -> None:
    """benchmark 子命令：只评检索，不调用生成模型。"""

    config_path = Path(args.config)
    config = load_config(config_path)
    documents = None
    if config.get("evidence_evaluation", False):
        # 区间标签与检索器必须使用同一批 Document，不能在两次加载间混入变更。
        verify_frozen_dataset(config_path)
        documents = load_documents(project_path(config_path, str(config["docs_path"])))
        retriever = build_retriever_from_documents(documents, config)
    else:
        retriever, config = build_retriever(config_path)
    cases = load_evaluation_cases(
        project_path(config_path, str(config["eval_path"]))
    )
    top_k = int(config.get("top_k", 3))
    report = evaluate_retrieval(retriever, cases, top_k=top_k, documents=documents)
    report["config"] = config
    report["dataset"] = verify_frozen_dataset(config_path)
    report["runtime"] = runtime_fingerprint()
    if documents is not None:
        report["input_sha256"] = {
            "config": text_sha256(config_path),
            "eval": text_sha256(project_path(config_path, str(config["eval_path"]))),
        }
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果已写入: {Path(args.output).resolve()}")


def run_query(args: argparse.Namespace) -> None:
    """query 子命令：打印 Top-K，适合人工观察某个问题为什么检索成这样。"""

    config_path = Path(args.config)
    retriever, config = build_retriever(config_path)
    results = retriever.search(args.question, top_k=int(config.get("top_k", 3)))
    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] {result.chunk.chunk_id} score={result.score:.4f}")
        print(result.chunk.text[:500])
        print()


def run_answer(args: argparse.Namespace) -> None:
    """answer 子命令：检索一次并调用指定模型生成单题答案。"""

    config_path = Path(args.config)
    retriever, config = build_retriever(config_path)
    results = retriever.search(args.question, top_k=int(config.get("top_k", 3)))
    # DeepSeek 走云端 API；Ollama 走本机服务，但都返回相同 GenerationResult。
    if args.provider == "deepseek":
        generator = DeepSeekGenerator(
            model=args.model or "deepseek-v4-flash",
            base_url=args.base_url or "https://api.deepseek.com",
        )
    else:
        generator = OllamaGenerator(
            model=args.model or "qwen2.5:1.5b",
            base_url=args.base_url or "http://127.0.0.1:11434",
        )
    generation = generator.generate(args.question, results)
    print(generation.text)
    print(
        "\nAPI 用量: "
        f"{generation.stats.total_tokens} tokens, "
        f"{generation.stats.latency_seconds:.2f}s"
    )
    print("\n检索来源:")
    for result in results:
        print(f"- {result.chunk.chunk_id} (score={result.score:.4f})")


def run_answer_benchmark(args: argparse.Namespace) -> None:
    """answer-benchmark 子命令：批量生成、Judge 评分并保存端到端报告。"""

    config_path = Path(args.config)
    retriever, config = build_retriever(config_path)
    cases = load_evaluation_cases(
        project_path(config_path, str(config["eval_path"]))
    )
    if args.case_id:
        # --case-id 可重复传入，用于只回归之前失败的少量样本。
        requested_ids = set(args.case_id)
        available_ids = {case.case_id for case in cases}
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            raise ValueError(
                "unknown --case-id: " + ", ".join(missing_ids)
            )
        cases = [case for case in cases if case.case_id in requested_ids]
    if args.limit is not None:
        # --limit 适合先做低成本 smoke test，再运行完整评测集。
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]

    # 每题会调用 generator 和 judge 各一次，所以总 API 次数是题数的两倍。
    generator = DeepSeekGenerator(
        model=args.generator_model,
        base_url=args.base_url,
    )
    judge = DeepSeekAnswerJudge(
        model=args.judge_model,
        base_url=args.base_url,
    )
    print(f"将评测 {len(cases)} 题，预计调用 DeepSeek {len(cases) * 2} 次。")
    report = evaluate_answers(
        retriever,
        cases,
        generator,
        judge,
        top_k=int(config.get("top_k", 3)),
    )
    report["config"] = config
    report["models"] = {
        "generator": args.generator_model,
        "judge": args.judge_model,
    }
    report["dataset"] = verify_frozen_dataset(config_path)
    report["runtime"] = runtime_fingerprint()
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果已写入: {Path(args.output).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    """声明四个子命令及其参数，并用 handler 绑定对应执行函数。"""

    parser = argparse.ArgumentParser(description="DomainRAG retrieval benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="run retrieval evaluation")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.set_defaults(handler=run_benchmark)

    query = subparsers.add_parser("query", help="inspect retrieved chunks")
    query.add_argument("--config", required=True)
    query.add_argument("--question", required=True)
    query.set_defaults(handler=run_query)

    answer = subparsers.add_parser("answer", help="retrieve and generate an answer")
    answer.add_argument("--config", required=True)
    answer.add_argument("--question", required=True)
    answer.add_argument(
        "--provider",
        choices=("deepseek", "ollama"),
        default="deepseek",
    )
    answer.add_argument("--model", default=None)
    answer.add_argument("--base-url", default=None)
    answer.set_defaults(handler=run_answer)

    answer_benchmark = subparsers.add_parser(
        "answer-benchmark",
        help="generate answers and evaluate correctness, faithfulness, and citations",
    )
    answer_benchmark.add_argument("--config", required=True)
    answer_benchmark.add_argument("--output", required=True)
    answer_benchmark.add_argument("--limit", type=int, default=None)
    answer_benchmark.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="evaluate only this case id; repeat the option for multiple cases",
    )
    answer_benchmark.add_argument(
        "--generator-model",
        default="deepseek-v4-flash",
    )
    answer_benchmark.add_argument(
        "--judge-model",
        default="deepseek-v4-flash",
    )
    answer_benchmark.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
    )
    answer_benchmark.set_defaults(handler=run_answer_benchmark)
    return parser


def main() -> None:
    """统一设置中文输出编码，解析参数并分发给所选子命令。"""

    # Windows 终端默认编码可能不是 UTF-8，显式设置可避免中文报告乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    # set_defaults(handler=...) 让这里无需再写一大段 if/elif 判断命令名。
    args.handler(args)


if __name__ == "__main__":
    main()
