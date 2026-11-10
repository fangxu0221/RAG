"""组织答案评测，并提供统一的检索评测入口。"""

from __future__ import annotations

import re
from collections.abc import Callable
from statistics import mean

from ..domain.protocols import AnswerGenerator, AnswerJudge, ApiCallStats, CitationAssessment
from ..evaluation import evaluate_retrieval
from ..failures import failure_counts, retrieval_failures
from ..models import EvaluationCase, SearchResult


CITATION_PATTERN = re.compile(r"\[(?:来源\s*[:：]\s*)?(\d+)\]")


def evaluate_citations(
    answer: str,
    results: list[SearchResult],
    relevant_sources: tuple[str, ...],
    citation_assessments: tuple[CitationAssessment, ...] | None = None,
) -> dict[str, object]:
    """解析答案中的引用，并计算 Validity、Precision、Recall、F1 与 Accuracy。

    - Validity：引用编号是否真的存在，例如只有 2 个 Chunk 却引用 [3] 就无效。
    - Precision：有效引用中，有多少 Chunk 真正支持回答结论。
    - Recall：标准相关来源中，有多少被“支持回答的引用”覆盖。
    - Accuracy：Validity × Citation F1，同时惩罚编号错误与证据质量问题。
    """

    # 正则兼容 [1] 与 [来源：1]；保留重复项用于正确计算编号有效率。
    cited_indices = [int(value) for value in CITATION_PATTERN.findall(answer)]
    valid_indices = [index for index in cited_indices if 1 <= index <= len(results)]
    invalid_indices = [index for index in cited_indices if index not in valid_indices]
    # Precision 按不同证据块计算，因此先去重，重复写 [1][1] 不能重复获益或受罚。
    unique_valid_indices = list(dict.fromkeys(valid_indices))
    cited_sources = {results[index - 1].chunk.source for index in unique_valid_indices}
    relevant = set(relevant_sources)

    if citation_assessments is None:
        # 没有 Chunk 级 Judge 判断时，退化为 source 级标签推断；精度较粗但兼容旧调用。
        assessment_by_index = {
            index: CitationAssessment(
                index=index,
                supported=results[index - 1].chunk.source in relevant,
                reason="由文档级正确来源标签推断",
            )
            for index in unique_valid_indices
        }
    else:
        assessment_by_index = {
            assessment.index: assessment for assessment in citation_assessments
        }
    # Judge 漏评某个已引用编号时保守地视为不支持，避免指标虚高。
    resolved_assessments = {
        index: assessment_by_index.get(
            index,
            CitationAssessment(index, False, "Judge 未返回该引用的支持判断"),
        )
        for index in unique_valid_indices
    }
    supported_indices = [
        index
        for index in unique_valid_indices
        if resolved_assessments[index].supported
    ]
    supported_sources = {
        results[index - 1].chunk.source for index in supported_indices
    }
    matched = supported_sources.intersection(relevant)

    # 编号口诀：Validity 看“编号对不对”。
    validity = len(valid_indices) / len(cited_indices) if cited_indices else 0.0
    # Precision 看“引用是否多引了不支持结论的 Chunk”。
    precision = (
        len(supported_indices) / len(unique_valid_indices)
        if unique_valid_indices
        else 0.0
    )
    # Recall 看“正确来源是否漏引”。当前标签是 source 级，不是 Chunk 数量级。
    recall = len(matched) / len(relevant) if relevant else 0.0
    citation_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "cited_indices": cited_indices,
        "invalid_indices": invalid_indices,
        "cited_sources": sorted(cited_sources),
        "supported_indices": supported_indices,
        "unsupported_indices": [
            index for index in unique_valid_indices if index not in supported_indices
        ],
        "citation_assessments": [
            {
                "index": index,
                "chunk_id": results[index - 1].chunk.chunk_id,
                "source": results[index - 1].chunk.source,
                "supported": resolved_assessments[index].supported,
                "reason": resolved_assessments[index].reason,
            }
            for index in unique_valid_indices
        ],
        "citation_validity": validity,
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": citation_f1,
        "citation_accuracy": validity * citation_f1,
    }


def _stats_dict(stats: ApiCallStats) -> dict[str, int | float]:
    """将不可 JSON 序列化的 dataclass 转为报告字典，并统一延迟精度。"""

    return {
        "latency_seconds": round(stats.latency_seconds, 4),
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.total_tokens,
    }


def _combined_stats(*stats_items: ApiCallStats) -> ApiCallStats:
    """合并一次生成调用与一次 Judge 调用的耗时和 Token。"""

    return ApiCallStats(
        latency_seconds=sum(item.latency_seconds for item in stats_items),
        prompt_tokens=sum(item.prompt_tokens for item in stats_items),
        completion_tokens=sum(item.completion_tokens for item in stats_items),
        total_tokens=sum(item.total_tokens for item in stats_items),
    )


SCORE_FIELDS = (
    "correctness",
    "faithfulness",
    "citation_validity",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "citation_accuracy",
)


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """对一组题目计算指标均值和 API 总量/平均延迟。"""

    api_totals = [row["api"]["total"] for row in rows]
    # 每题固定调用两次模型：一次生成答案，一次 Judge 评分。
    api_calls = len(rows) * 2
    latency_seconds = sum(float(item["latency_seconds"]) for item in api_totals)
    summary: dict[str, object] = {
        "cases": len(rows),
        **{
            field: (mean(float(row[field]) for row in rows if row[field] is not None)
                    if any(row[field] is not None for row in rows) else None)
            for field in SCORE_FIELDS
        },
    }
    summary["metric_case_counts"] = {
        field: sum(row[field] is not None for row in rows) for field in SCORE_FIELDS
    }
    unknown = [row for row in rows if not row["answerable"]]
    known = [row for row in rows if row["answerable"]]
    judged_unknown = [row for row in unknown if row["abstained"] is not None]
    judged_known = [row for row in known if row["abstained"] is not None]
    refused = [row for row in rows if row["abstained"] is True]
    summary["refusal"] = {
        "answerable_cases": len(known), "unanswerable_cases": len(unknown),
        "judged_unanswerable_cases": len(judged_unknown),
        "judged_answerable_cases": len(judged_known),
        "refusal_recall": mean(float(row["abstained"]) for row in judged_unknown) if judged_unknown else None,
        "false_refusal_rate": mean(float(row["abstained"]) for row in judged_known) if judged_known else None,
        "refusal_precision": mean(float(not row["answerable"]) for row in refused) if refused else None,
        "unsupported_answer_rate": mean(float(not row["abstained"]) for row in judged_unknown) if judged_unknown else None,
    }
    summary["api"] = {
        "calls": api_calls,
        "prompt_tokens": sum(int(item["prompt_tokens"]) for item in api_totals),
        "completion_tokens": sum(
            int(item["completion_tokens"]) for item in api_totals
        ),
        "total_tokens": sum(int(item["total_tokens"]) for item in api_totals),
        "latency_seconds": round(latency_seconds, 4),
        "average_call_latency_seconds": round(
            latency_seconds / api_calls if api_calls else 0.0,
            4,
        ),
        "average_case_latency_seconds": round(
            latency_seconds / len(rows) if rows else 0.0,
            4,
        ),
    }
    return summary


def evaluate_answers(
    retriever: Retriever,
    cases: list[EvaluationCase],
    generator: AnswerGenerator,
    judge: AnswerJudge,
    top_k: int,
    on_case: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """运行完整答案评测，并输出总览、分类统计、失败题和逐题详情。"""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("answer evaluation requires at least one case")
    # 在任何检索/API 调用之前统一校验，避免花费 Token 后才发现没有参考答案。
    missing_answers = [case.case_id for case in cases if not case.expected_answer]
    if missing_answers:
        raise ValueError(
            "answer evaluation requires expected_answer for every case; "
            f"missing: {', '.join(missing_answers)}"
        )

    rows: list[dict[str, object]] = []
    for case in cases:
        # 一道题的完整链路：检索一次，生成与 Judge 共享同一份排序结果。
        results = retriever.search(case.question, top_k=top_k)
        generation = generator.generate(case.question, results)
        answer = generation.text
        judgement = judge.judge(case, answer, results)
        if not case.answerable and judgement.abstained is None:
            raise ValueError("unanswerable cases require an explicit Judge abstained flag")
        citations = evaluate_citations(
            answer,
            results,
            case.relevant_sources,
            judgement.citation_assessments,
        )
        # 无答案题的引用指标不适用；保留原始引用诊断，但不混入可回答题均值。
        if not case.answerable:
            for field in SCORE_FIELDS:
                if field.startswith("citation_"):
                    citations[field] = None
        failure_types = retrieval_failures(case, results)
        if judgement.correctness < 2:
            failure_types.append("incorrect_answer")
        if judgement.faithfulness < 2:
            failure_types.append("unsupported_claim")
        if case.answerable and judgement.abstained is True:
            failure_types.append("unnecessary_refusal")
        if not case.answerable and judgement.abstained is False:
            failure_types.append("unsupported_answer")
        if citations["invalid_indices"]:
            failure_types.append("invalid_citation")
        if citations["unsupported_indices"]:
            failure_types.append("unsupported_citation")
        if case.answerable and citations["citation_recall"] < 1:
            failure_types.append("missing_citation_source")
        total_stats = _combined_stats(generation.stats, judgement.stats)
        rows.append(
            {
                "id": case.case_id,
                "question": case.question,
                "answerable": case.answerable,
                "abstained": judgement.abstained,
                "failure_types": failure_types,
                "expected_answer": case.expected_answer,
                "answer": answer,
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": result.chunk.chunk_id,
                        "source": result.chunk.source,
                        "score": round(result.score, 6),
                        "text": result.chunk.text,
                    }
                    for rank, result in enumerate(results, start=1)
                ],
                # Judge 原始值为 0/1/2；除以 2 统一成报告中的 0~1 指标。
                "correctness": judgement.correctness / 2,
                "faithfulness": judgement.faithfulness / 2,
                "judge_reason": judgement.reason,
                **citations,
                "api": {
                    "generation": _stats_dict(generation.stats),
                    "judge": _stats_dict(judgement.stats),
                    "total": _stats_dict(total_stats),
                },
                "metadata": case.metadata,
            }
        )
        if on_case is not None:
            on_case(rows[-1])

    return summarize_answers(rows)


def summarize_answers(rows: list[dict[str, object]]) -> dict[str, object]:
    """汇总已完成逐题结果，也供检查点恢复工具使用。"""
    # metadata.question_type 用于观察直接问法、同义改写、干扰题等类别的差异。
    rows_by_type: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        metadata = row["metadata"]
        question_type = str(metadata.get("question_type", "unknown"))
        rows_by_type.setdefault(question_type, []).append(row)

    return {
        "summary": _summarize_rows(rows),
        "by_answerability": {
            label: _summarize_rows([row for row in rows if row["answerable"] == flag])
            for label, flag in (("answerable", True), ("unanswerable", False))
        },
        "failure_type_counts": failure_counts(rows),
        "by_question_type": {
            question_type: _summarize_rows(type_rows)
            for question_type, type_rows in sorted(rows_by_type.items())
        },
        # 任一指标未满分就进入失败清单，便于用 --case-id 做最小回归。
        "failure_cases": [
            {
                "id": row["id"],
                "question_type": row["metadata"].get("question_type", "unknown"),
                "correctness": row["correctness"],
                "faithfulness": row["faithfulness"],
                "citation_accuracy": row["citation_accuracy"],
                "unsupported_indices": row["unsupported_indices"],
                "judge_reason": row["judge_reason"],
                "failure_types": row["failure_types"],
            }
            for row in rows
            if row["failure_types"]
        ],
        "cases": rows,
    }
