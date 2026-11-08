"""离线检索评测：计算 Hit Rate@K、Recall@K 和 MRR。

默认保留 source 文档级指标；传入同一快照的 documents 时增加区间覆盖的 Evidence 指标。
不调用大模型，也不评价最终回答质量。
"""

from __future__ import annotations

from statistics import mean

from .models import Document, EvaluationCase
from .evidence import EvidenceGroundTruth
from .retriever import Retriever
from .failures import failure_counts, retrieval_failures


def _summary(rows: list[dict[str, object]], top_k: int) -> dict[str, object]:
    applicable = [row for row in rows if row["answerable"]]
    summary = {
        "cases": len(rows),
        "answerable_cases": len(applicable),
        "unanswerable_cases": len(rows) - len(applicable),
        f"hit_rate@{top_k}": mean(float(row["hit"]) for row in applicable) if applicable else None,
        f"recall@{top_k}": mean(float(row["recall"]) for row in applicable) if applicable else None,
        "mrr": mean(float(row["reciprocal_rank"]) for row in applicable) if applicable else None,
    }
    if rows and "evidence_status" in rows[0]:
        summary["evidence_evaluated_cases"] = len(applicable)
        for key in (f"evidence_hit@{top_k}", f"evidence_recall@{top_k}",
                    "evidence_mrr", f"sufficient_evidence@{top_k}"):
            summary[key] = mean(float(row[key]) for row in applicable) if applicable else None
    return summary


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[EvaluationCase],
    top_k: int,
    *,
    documents: list[Document] | None = None,
) -> dict[str, object]:
    """逐题检索并汇总三项经典排名指标。"""

    if top_k <= 0 or not cases:
        raise ValueError("retrieval evaluation requires positive top_k and nonempty cases")
    # 先验证全部标签；旧题集不传 documents，保持原报告和评测行为。
    ground_truth = EvidenceGroundTruth(documents, cases) if documents is not None else None
    rows: list[dict[str, object]] = []

    for case in cases:
        # 每道题只检索一次，后续所有指标都复用同一份排序结果。
        results = retriever.search(case.question, top_k=top_k)
        relevant = set(case.relevant_sources)
        retrieved_sources = [result.chunk.source for result in results]
        # 同一文档可能切出多个 Chunk；Recall 的分母是正确文档数，需先按 source 去重。
        unique_retrieved = list(dict.fromkeys(retrieved_sources))

        matched_sources = relevant.intersection(unique_retrieved)
        # Recall@K = Top-K 中检索到的正确文档数 / 该题全部正确文档数。
        recall = len(matched_sources) / len(relevant) if relevant else 0.0
        # MRR 只关心“第一个正确结果”排第几：第 1 名得 1，第 2 名得 1/2，以此类推。
        first_rank = next(
            (rank for rank, source in enumerate(retrieved_sources, start=1) if source in relevant),
            None,
        )
        reciprocal_rank = 1 / first_rank if first_rank else 0.0

        rows.append(
            {
                "id": case.case_id,
                "question": case.question,
                "answerable": case.answerable,
                "metadata": case.metadata,
                "failure_types": retrieval_failures(case, results),
                "answer_evaluation_status": "not_run",
                "relevant_sources": list(case.relevant_sources),
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": result.chunk.chunk_id,
                        "source": result.chunk.source,
                        "score": round(result.score, 6),
                        "preview": result.chunk.text[:160],
                    }
                    for rank, result in enumerate(results, start=1)
                ],
                # Hit 只回答“Top-K 里是否至少出现一个正确文档”。
                "hit": bool(first_rank) if case.answerable else None,
                "recall": recall if case.answerable else None,
                "reciprocal_rank": reciprocal_rank if case.answerable else None,
            }
        )
        if ground_truth is not None:
            rows[-1].update(ground_truth.score(case, results, top_k))
            for item, result in zip(rows[-1]["retrieved"], results):
                item.update(start=result.chunk.start, end=result.chunk.end)

    # 报告总分是所有评测题逐题分数的算术平均。
    types = sorted({str(row["metadata"].get("question_type", "unknown")) for row in rows})
    report = {
        "summary": _summary(rows, top_k),
        "by_question_type": {
            kind: _summary([row for row in rows if str(row["metadata"].get("question_type", "unknown")) == kind], top_k)
            for kind in types
        },
        "failure_type_counts": failure_counts(rows),
        "failure_cases": [row for row in rows if row["failure_types"]],
        "cases": rows,
    }
    if ground_truth is not None:
        report["evidence_protocol"] = ground_truth.protocol()
        report["source_hit_evidence_miss_cases"] = [
            row["id"] for row in rows
            if row["hit"] and not row[f"evidence_hit@{top_k}"]
        ]
        unknown = [row for row in rows if not row["answerable"]]
        report["unanswerable_retrieval"] = {
            "cases": len(unknown),
            "with_results": sum(bool(row["retrieved"]) for row in unknown),
            "case_ids": [row["id"] for row in unknown],
            "note": "Retrieval alone does not establish successful refusal or answer quality.",
        }
    return report
