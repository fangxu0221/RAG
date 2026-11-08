"""以可控坐标验证证据指标；不调用模型，也不根据 Dev 检索成绩改标签。"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from domainrag.chunking import chunk_document
from domainrag.cli import run_benchmark
from domainrag.evaluation import evaluate_retrieval
from domainrag.evidence import EvidenceGroundTruth
from domainrag.frozen import text_sha256
from domainrag.io import load_documents, load_evaluation_cases
from domainrag.models import Chunk, Document, EvaluationCase, SearchResult
from domainrag.text import normalize_text


class StaticRetriever:
    def __init__(self, chunks):
        self.results = [SearchResult(chunk, 1.0) for chunk in chunks]
        self.calls = 0

    def search(self, query, top_k=3):
        self.calls += 1
        return self.results[:top_k]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("a.md", "abcdefghij")

    def chunk(self, start, end, source="a.md", text=None):
        return Chunk(f"{source}#{start}-{end}", source,
                     self.doc.text[start:end] if text is None else text, start, end)

    def case(self, *quotes, items=None, case_id="one"):
        return EvaluationCase(case_id, case_id, ("a.md",), metadata={
            "evidence": items if items is not None else [
                {"source": "a.md", "quote": quote} for quote in quotes
            ],
        })

    def report(self, case, chunks, k=3):
        retriever = StaticRetriever(chunks)
        report = evaluate_retrieval(retriever, [case], k, documents=[self.doc])
        self.assertEqual(retriever.calls, 1)
        return report

    def test_right_source_wrong_chunk_is_not_evidence(self):
        report = self.report(self.case("hij"), [self.chunk(0, 4)])
        row = report["cases"][0]
        self.assertTrue(row["hit"])
        self.assertFalse(row["evidence_hit@3"])
        self.assertEqual(report["source_hit_evidence_miss_cases"], ["one"])

    def test_partial_quote_does_not_count(self):
        row = self.report(self.case("cdef"), [self.chunk(0, 4)])["cases"][0]
        self.assertEqual(row["evidence_recall@3"], 0)

    def test_two_necessary_evidence_groups_partial_and_complete(self):
        case = self.case("ab", "ij")
        partial = self.report(case, [self.chunk(0, 3)])["cases"][0]
        self.assertTrue(partial["evidence_hit@3"])
        self.assertEqual(partial["evidence_recall@3"], .5)
        self.assertFalse(partial["sufficient_evidence@3"])
        complete = self.report(case, [self.chunk(3, 5), self.chunk(8, 10), self.chunk(0, 2)])["cases"][0]
        self.assertEqual(complete["evidence_recall@3"], 1)
        self.assertEqual(complete["evidence_mrr"], .5)
        self.assertTrue(complete["sufficient_evidence@3"])

    def test_duplicate_labels_and_retrieved_chunks_do_not_change_denominator(self):
        row = self.report(self.case("ab", "ab", "ij"), [self.chunk(0, 2)] * 3)["cases"][0]
        self.assertEqual(len(row["evidence_groups"]), 2)
        self.assertEqual(row["evidence_recall@3"], .5)

    def test_explicit_or_alternatives_and_duplicate_groups(self):
        items = [
            {"source": "a.md", "quote": q, "group_id": g}
            for g in ("first", "duplicate") for q in ("ab", "ij")
        ]
        row = self.report(self.case(items=items), [self.chunk(8, 10)])["cases"][0]
        self.assertEqual(len(row["evidence_groups"]), 1)
        self.assertTrue(row["sufficient_evidence@3"])

    def test_repeated_quote_requires_review_and_any_occurrence_is_or(self):
        self.doc = Document("a.md", "abc----abc")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.report(self.case("abc"), [])
        item = {"source": "a.md", "quote": "abc", "match_policy": "any_occurrence"}
        row = self.report(self.case(items=[item, item]), [self.chunk(7, 10)])["cases"][0]
        self.assertEqual(len(row["evidence_groups"]), 1)
        self.assertEqual(len(row["evidence_groups"][0]["alternatives"][0]["normalized_spans"]), 2)
        self.assertEqual(row["evidence_recall@3"], 1)

    def test_adjacent_chunks_cover_quote_at_completion_rank(self):
        row = self.report(self.case("cdefgh"), [self.chunk(0, 5), self.chunk(5, 10)])["cases"][0]
        self.assertTrue(row["sufficient_evidence@3"])
        self.assertEqual(row["evidence_mrr"], .5)
        limited = self.report(self.case("cdefgh"), [self.chunk(0, 5), self.chunk(5, 10)], k=1)["cases"][0]
        self.assertFalse(limited["evidence_hit@1"])

    def test_overlaps_merge_but_gaps_do_not(self):
        full = self.report(self.case("cdefgh"), [self.chunk(0, 6), self.chunk(4, 9)])["cases"][0]
        self.assertEqual(full["evidence_recall@3"], 1)
        gap = self.report(self.case("cdefgh"), [self.chunk(0, 4), self.chunk(5, 9)])["cases"][0]
        self.assertEqual(gap["evidence_recall@3"], 0)

    def test_no_union_across_sources(self):
        docs = [self.doc, Document("b.md", self.doc.text)]
        results = StaticRetriever([self.chunk(0, 5), self.chunk(5, 10, "b.md")])
        report = evaluate_retrieval(results, [self.case("cdefgh")], 3, documents=docs)
        self.assertFalse(report["cases"][0]["evidence_hit@3"])

    def test_or_across_sources_still_requires_other_and_group(self):
        case = EvaluationCase("one", "one", ("a.md", "b.md"), metadata={"evidence": [
            {"source": "a.md", "quote": "ab", "group_id": "fact"},
            {"source": "b.md", "quote": "XY", "group_id": "fact"},
            {"source": "a.md", "quote": "ij", "group_id": "other"},
        ]})
        retriever = StaticRetriever([Chunk("b#0", "b.md", "XY", 0, 2), self.chunk(8, 10)])
        report = evaluate_retrieval(retriever, [case], 3, documents=[self.doc, Document("b.md", "XYZ")])
        self.assertTrue(report["cases"][0]["sufficient_evidence@3"])

    def test_missing_or_unlocatable_evidence_fails_before_any_search(self):
        invalid = [self.case(items=[]), self.case("not present"), self.case(items=[
            {"source": "missing.md", "quote": "abc"}
        ])]
        for case in invalid:
            retriever = StaticRetriever([])
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluate_retrieval(retriever, [self.case("ab", case_id="valid"), case], 3, documents=[self.doc])
            self.assertEqual(retriever.calls, 0)

    def test_normalization_matches_real_chunker_coordinates(self):
        doc = Document("a.md", "  ＡＢＣ\n\t退票５％。 后文  ")
        case = self.case("ＡＢＣ\n\t退票５％。")
        chunks = chunk_document(doc, chunk_size=7, chunk_overlap=2)
        report = evaluate_retrieval(StaticRetriever(chunks), [case], 4, documents=[doc])
        self.assertTrue(report["cases"][0]["sufficient_evidence@4"])
        self.assertEqual(report["cases"][0]["evidence_mrr"], .5)

    def test_corrupt_chunk_coordinates_or_text_fail(self):
        bad = [self.chunk(0, 11), self.chunk(0, 3, text="hij"), self.chunk(-1, 3),
               self.chunk(0, 2, "missing.md")]
        for chunk in bad:
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, "invalid chunk"):
                self.report(self.case("ab"), [chunk])

    def test_overlapping_group_definitions_are_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "different necessary groups"):
            self.report(self.case(items=[
                {"source": "a.md", "quote": "ab", "group_id": "x"},
                {"source": "a.md", "quote": "ab", "group_id": "y"},
                {"source": "a.md", "quote": "ij", "group_id": "y"},
            ]), [])

    def test_unanswerable_excluded_and_all_unknown_means_null(self):
        unknown = EvaluationCase("unknown", "unknown", (), answerable=False, metadata={
            "evidence": [], "unanswerable_reason": "outside corpus",
        })
        for cases in ([self.case("ab"), unknown], [unknown]):
            report = evaluate_retrieval(StaticRetriever([self.chunk(0, 3)]), cases, 3, documents=[self.doc])
            self.assertEqual(report["unanswerable_retrieval"]["cases"], 1)
            self.assertIsNone(report["cases"][-1]["evidence_recall@3"])
            expected = 1 if len(cases) == 2 else None
            for key in ("evidence_hit@3", "evidence_recall@3", "evidence_mrr", "sufficient_evidence@3"):
                self.assertEqual(report["summary"][key], expected)

    def test_bad_unanswerable_labels_rejected(self):
        unknown = EvaluationCase("x", "x", (), answerable=False, metadata={"evidence": []})
        with self.assertRaisesRegex(ValueError, "reason"):
            EvidenceGroundTruth([self.doc], [unknown])

    def test_source_metrics_and_old_report_unchanged_when_opt_in(self):
        case = self.case("ij")
        old = evaluate_retrieval(StaticRetriever([self.chunk(0, 2)]), [case], 3)
        new = self.report(case, [self.chunk(0, 2)])
        for key, value in old["summary"].items():
            self.assertEqual(new["summary"][key], value)
        self.assertNotIn("evidence_protocol", old)
        self.assertNotIn("evidence_status", old["cases"][0])
        for key in ("hit", "recall", "reciprocal_rank", "failure_types"):
            self.assertEqual(old["cases"][0][key], new["cases"][0][key])

    def test_cli_opt_in_and_fingerprints(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "docs").mkdir()
            (root / "docs/a.md").write_text("退票规定", encoding="utf-8")
            (root / "eval.jsonl").write_text(json.dumps({
                "id": "one", "question": "退票", "relevant_sources": ["a.md"],
                "metadata": {"evidence": [{"source": "a.md", "quote": "退票规定"}]},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config = root / "configs/dev.json"
            config.write_text(json.dumps({"docs_path": "docs", "eval_path": "eval.jsonl",
                "retriever": "bm25", "chunk_size": 420, "chunk_overlap": 80,
                "top_k": 3, "evidence_evaluation": True}), encoding="utf-8")
            output = root / "result.json"
            with redirect_stdout(StringIO()):
                run_benchmark(argparse.Namespace(config=str(config), output=str(output)))
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["evidence_recall@3"], 1)
            self.assertEqual(report["input_sha256"]["eval"], text_sha256(root / "eval.jsonl"))
            self.assertEqual(report["cases"][0]["retrieved"][0]["start"], 0)


class RailwayAnnotationTests(unittest.TestCase):
    def test_existing_ground_truth_and_config_integrity(self):
        root = Path(__file__).resolve().parents[1]
        corpus = root / "data/railway"
        cases = load_evaluation_cases(corpus / "eval_dev.jsonl")
        docs = load_documents(corpus)
        gt = EvidenceGroundTruth(docs, cases)
        validation = json.loads((corpus / "eval_dev.validation.json").read_text(encoding="utf-8"))
        self.assertEqual(text_sha256(corpus / "eval_dev.jsonl"), validation["dataset_sha256"])
        self.assertEqual(gt.corpus_sha256, validation["corpus_sha256"])
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case.answerable for case in cases), 8)
        self.assertEqual([len(gt.groups[case.case_id]) for case in cases], [1, 1, 2, 1, 1, 1, 3, 3, 0, 0])
        self.assertEqual(len(gt.groups["rail-dev-003"][0].alternatives[0].spans), 2)
        units = {u["unit_id"]: u for u in map(json.loads, (corpus / "units.jsonl").read_text(encoding="utf-8").splitlines())}
        raw_docs = {doc.source: doc.text for doc in docs}
        for case in cases:
            for item in case.metadata["evidence"]:
                self.assertIn(item["quote"], raw_docs[item["source"]])
                self.assertIn(item["quote"], units[item["unit_id"]]["text"])
                self.assertEqual(item["source"], units[item["unit_id"]]["source"])
        baseline = json.loads((root / "configs/bm25_baseline.json").read_text(encoding="utf-8"))
        dev = json.loads((root / "configs/railway_dev_bm25.json").read_text(encoding="utf-8"))
        for key in ("retriever", "chunk_size", "chunk_overlap", "top_k", "bm25"):
            self.assertEqual(baseline[key], dev[key])


if __name__ == "__main__":
    unittest.main()
