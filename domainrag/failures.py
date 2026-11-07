"""可复用的失败标签；它们是可观测症状，不是自动证明的因果归因。"""

from collections import Counter

from .models import EvaluationCase, SearchResult


def retrieval_failures(case: EvaluationCase, results: list[SearchResult]) -> list[str]:
    if not case.answerable:
        return []  # 无答案题的 Top-K 非空不等于生成器一定会编造。
    relevant = set(case.relevant_sources)
    matched = relevant.intersection(result.chunk.source for result in results)
    if not matched:
        return ["retrieval_miss"]
    if matched != relevant:
        return ["retrieval_partial"]
    return []


def failure_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(label for row in rows for label in row["failure_types"]).items()))
