"""确定性证据评测：AND 必要证据组 / OR 替代引文与位置，坐标均为规范化字符偏移。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .models import Document, EvaluationCase, SearchResult
from .text import normalize_text


@dataclass(frozen=True)
class EvidenceAlternative:
    source: str
    quote: str
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    alternatives: tuple[EvidenceAlternative, ...]


def _occurrences(text: str, quote: str) -> tuple[tuple[int, int], ...]:
    spans = []
    start = text.find(quote)
    while start >= 0:
        spans.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    return tuple(spans)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并相邻、重叠区间；重复 Chunk 不会增加覆盖。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class EvidenceGroundTruth:
    """先校验全部标注，再检索；不根据检索分数推断或修补 Ground Truth。"""

    def __init__(self, documents: list[Document], cases: list[EvaluationCase]) -> None:
        self.documents = {doc.source: normalize_text(doc.text) for doc in documents}
        if not documents or len(self.documents) != len(documents):
            raise ValueError("evidence corpus must be nonempty with unique sources")
        self.corpus_sha256 = {
            doc.source: hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            for doc in documents
        }
        self.groups: dict[str, tuple[EvidenceGroup, ...]] = {}
        questions: set[str] = set()
        for case in cases:
            question = normalize_text(case.question).casefold()
            if not question or question in questions or case.case_id in self.groups:
                raise ValueError(f"duplicate/empty evidence case: {case.case_id}")
            questions.add(question)
            self.groups[case.case_id] = self._compile(case)

    def _compile(self, case: EvaluationCase) -> tuple[EvidenceGroup, ...]:
        def fail(message: str) -> None:
            raise ValueError(f"{case.case_id}: {message}")

        items = case.metadata.get("evidence")
        if not isinstance(items, list):
            fail("metadata.evidence must be a list")
        if not case.answerable:
            if items or case.relevant_sources or not case.metadata.get("unanswerable_reason"):
                fail("unanswerable requires empty evidence/sources and a reason")
            return ()
        if not items or not case.relevant_sources:
            fail("answerable case requires evidence and sources")
        # 同组 OR，组间 AND；没有 group_id 时，每段去重后的引文各自必要。
        grouped: dict[tuple[str, object], dict[tuple[str, str], EvidenceAlternative]] = {}
        sources: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                fail(f"evidence[{index}] must be an object")
            source, raw_quote = item.get("source"), item.get("quote")
            if not isinstance(source, str) or source not in self.documents:
                fail(f"evidence[{index}] source not found: {source!r}")
            if not isinstance(raw_quote, str) or not normalize_text(raw_quote):
                fail(f"evidence[{index}] quote must be nonempty")
            quote = normalize_text(raw_quote)
            spans = _occurrences(self.documents[source], quote)
            if not spans:
                fail(f"evidence[{index}] quote not found in normalized source")
            policy = item.get("match_policy", "unique")
            if policy not in ("unique", "any_occurrence"):
                fail(f"evidence[{index}] unsupported match_policy")
            if len(spans) != 1 and policy != "any_occurrence":
                fail(f"evidence[{index}] ambiguous quote ({len(spans)} occurrences); review alternatives explicitly")
            sources.add(source)
            atom = (source, quote)
            if "group_id" in item:
                if not isinstance(item["group_id"], str) or not item["group_id"].strip():
                    fail(f"evidence[{index}] group_id must be nonempty")
                key = ("explicit", item["group_id"])
            else:
                key = ("single", atom)
            grouped.setdefault(key, {})[atom] = EvidenceAlternative(source, quote, spans)
        if sources != set(case.relevant_sources):
            fail("evidence sources differ from relevant_sources")
        groups: list[EvidenceGroup] = []
        seen_groups: set[frozenset[tuple[str, str]]] = set()
        used_atoms: set[tuple[str, str]] = set()
        for key, alternatives in grouped.items():
            signature = frozenset(alternatives)
            if signature in seen_groups:
                continue
            if used_atoms.intersection(signature):
                fail("same evidence belongs to different necessary groups; ambiguous AND/OR annotation")
            seen_groups.add(signature)
            used_atoms.update(signature)
            group_id = str(key[1]) if key[0] == "explicit" else f"evidence-{len(groups) + 1}"
            groups.append(EvidenceGroup(group_id, tuple(alternatives.values())))
        if len({group.group_id for group in groups}) != len(groups):
            fail("group_id collides with an implicit group name")
        return tuple(groups)

    def score(self, case: EvaluationCase, results: list[SearchResult], top_k: int) -> dict[str, object]:
        groups = self.groups[case.case_id]
        # 校验坐标和实际文本属于同一快照，避免 raw/normalized 偏移混用产生假命中。
        intervals: dict[str, list[tuple[int, int]]] = {}
        first_ranks: list[int | None] = [None] * len(groups)
        for rank, result in enumerate(results[:top_k], 1):
            chunk = result.chunk
            text = self.documents.get(chunk.source)
            if (text is None or type(chunk.start) is not int or type(chunk.end) is not int
                    or not 0 <= chunk.start < chunk.end <= len(text)
                    or not chunk.text or text[chunk.start:chunk.end].strip() != chunk.text):
                raise ValueError(f"{case.case_id}: invalid chunk coordinates/text: {chunk.chunk_id}")
            intervals.setdefault(chunk.source, []).append((chunk.start, chunk.end))
            intervals[chunk.source] = _merge(intervals[chunk.source])
            for index, group in enumerate(groups):
                if first_ranks[index] is None and any(
                    left <= start and end <= right
                    for alternative in group.alternatives
                    for start, end in alternative.spans
                    for left, right in intervals.get(alternative.source, [])
                ):
                    first_ranks[index] = rank
        if not case.answerable:
            return {
                "evidence_status": "not_applicable_unanswerable",
                f"evidence_hit@{top_k}": None, f"evidence_recall@{top_k}": None,
                "evidence_mrr": None, f"sufficient_evidence@{top_k}": None,
                "evidence_groups": [],
            }
        hits = sum(rank is not None for rank in first_ranks)
        first = min((rank for rank in first_ranks if rank is not None), default=None)
        return {
            "evidence_status": "valid",
            f"evidence_hit@{top_k}": hits > 0,
            f"evidence_recall@{top_k}": hits / len(groups),
            "evidence_mrr": 1 / first if first else 0.0,
            f"sufficient_evidence@{top_k}": hits == len(groups),
            "evidence_groups": [
                {
                    "group_id": group.group_id,
                    "first_covered_rank": first_ranks[index],
                    "alternatives": [
                        {"source": alt.source, "normalized_quote": alt.quote,
                         "normalized_spans": [list(span) for span in alt.spans]}
                        for alt in group.alternatives
                    ],
                }
                for index, group in enumerate(groups)
            ],
        }

    def protocol(self) -> dict[str, object]:
        return {
            "version": "normalized_interval_v1",
            "normalization": "domainrag.text.normalize_text: NFKC + whitespace collapse + strip",
            "coordinates": "half-open Python character offsets in normalized full document",
            "coverage": "entire quote interval contained in union of adjacent/overlapping chunk windows",
            "logic": "AND of deduplicated groups; OR of group alternatives and approved quote occurrences",
            "mrr": "reciprocal of earliest prefix rank completing any evidence group; capped at top_k",
            "invalid_annotations": "raise ValueError before any search",
            "unanswerable": "separate counts, all evidence metrics null and excluded from means",
            "corpus_sha256": self.corpus_sha256,
        }
