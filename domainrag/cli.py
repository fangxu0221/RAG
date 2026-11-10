"""命令行参数解析、任务分发和结果输出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .infrastructure.llm.answer_judge import DeepSeekAnswerJudge
from .application.evaluation_service import evaluate_retrieval, evaluate_answers
# 兼容原先从 cli 导入装配函数的脚本。
from .bootstrap import (
    build_bm25,
    build_embedding,
    build_retriever,
    build_retriever_from_documents,
    load_retrieval_snapshot,
    project_path,
)
from .frozen import runtime_fingerprint, text_sha256, verify_frozen_dataset
from .infrastructure.llm.generator import DeepSeekGenerator, OllamaGenerator
from .io import load_evaluation_cases, write_json


def run_benchmark(args: argparse.Namespace) -> None:
    """benchmark 子命令：只评检索，不调用生成模型。"""

    config_path = Path(args.config)
    config, snapshot = load_retrieval_snapshot(config_path)
    retriever = build_retriever_from_documents(snapshot, config)
    # Evidence 校验与检索索引复用同一批 Document。
    documents = snapshot if config.get("evidence_evaluation", False) else None
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
