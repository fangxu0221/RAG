"""冻结与拒答回归：只用合成题验证规则，不根据测试集成绩调参。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from domainrag.answer_evaluation import (
    CitationAssessment, DeepSeekAnswerJudge, SemanticJudgement,
    build_judge_prompt, evaluate_answers,
)
from domainrag.evaluation import evaluate_retrieval
from domainrag.frozen import text_sha256, verify_frozen_dataset
from domainrag.generator import GenerationResult, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from domainrag.io import load_evaluation_cases
from domainrag.models import Chunk, EvaluationCase, SearchResult


class StaticRetriever:
    def search(self, query, top_k=3):
        return [SearchResult(Chunk("a#0", "a.md", "alpha", 0, 5), 1.0)]


class Generator:
    def generate(self, question, results):
        return GenerationResult("固定输出")


class Judge:
    def __init__(self, abstained):
        self.abstained = abstained

    def judge(self, case, answer, results):
        return SemanticJudgement(2, 2, "synthetic", (CitationAssessment(1, True, "ok"),),
                                 abstained=self.abstained[case.case_id])


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.known = EvaluationCase("known", "alpha?", ("a.md",), "alpha")
        self.unknown = EvaluationCase("unknown", "missing?", (), "资料不足", answerable=False)

    def test_unknown_retrieval_excluded_instead_of_counting_as_miss(self):
        report = evaluate_retrieval(StaticRetriever(), [self.known, self.unknown], 3)
        self.assertEqual(report["summary"]["recall@3"], 1.0)
        self.assertEqual(report["summary"]["answerable_cases"], 1)
        self.assertIsNone(report["cases"][1]["recall"])
        self.assertEqual(report["cases"][1]["failure_types"], [])
        self.assertEqual(report["cases"][1]["answer_evaluation_status"], "not_run")

    def test_all_unknown_retrieval_has_null_metrics(self):
        self.assertIsNone(evaluate_retrieval(StaticRetriever(), [self.unknown], 3)["summary"]["mrr"])

    def test_partial_and_missed_sources_have_distinct_labels(self):
        cases = [EvaluationCase("partial", "?", ("a.md", "b.md")),
                 EvaluationCase("miss", "?", ("b.md",))]
        rows = evaluate_retrieval(StaticRetriever(), cases, 3)["cases"]
        self.assertEqual(rows[0]["failure_types"], ["retrieval_partial"])
        self.assertEqual(rows[1]["failure_types"], ["retrieval_miss"])

    def test_correct_refusal_without_citations_is_not_a_failure(self):
        saved = []
        report = evaluate_answers(StaticRetriever(), [self.unknown], Generator(),
                                  Judge({"unknown": True}), 3, on_case=saved.append)
        self.assertEqual(report["failure_cases"], [])
        self.assertEqual(report["summary"]["refusal"]["refusal_recall"], 1)
        self.assertIsNone(report["summary"]["citation_accuracy"])
        self.assertEqual(report["summary"]["metric_case_counts"]["citation_accuracy"], 0)
        self.assertEqual(len(saved), 1)

    def test_mixed_denominators_and_false_refusal(self):
        report = evaluate_answers(StaticRetriever(), [self.known, self.unknown], Generator(),
                                  Judge({"known": True, "unknown": False}), 3)
        refusal = report["summary"]["refusal"]
        self.assertEqual(refusal["refusal_recall"], 0)
        self.assertEqual(refusal["false_refusal_rate"], 1)
        self.assertEqual(refusal["refusal_precision"], 0)
        self.assertEqual(refusal["unsupported_answer_rate"], 1)
        self.assertEqual(report["summary"]["metric_case_counts"]["citation_accuracy"], 1)
        self.assertIn("unnecessary_refusal", report["cases"][0]["failure_types"])
        self.assertIn("unsupported_answer", report["cases"][1]["failure_types"])

    def test_unmeasured_refusal_not_assumed_success(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            evaluate_answers(StaticRetriever(), [self.unknown], Generator(), Judge({"unknown": None}), 3)

    def test_label_is_only_in_judge_prompt(self):
        self.assertIn("False", build_judge_prompt(self.unknown, "无法确定", []))
        from domainrag.generator import build_user_prompt
        self.assertNotIn("参考答案", build_user_prompt(self.unknown.question, []))

    def test_jsonl_rejects_ambiguous_answerability(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"eval.jsonl"
            for answerable, sources in [("false", []), (False, ["a.md"]), (True, [])]:
                path.write_text(json.dumps({"id": "x", "question": "?", "relevant_sources": sources,
                                            "answerable": answerable}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_evaluation_cases(path)

    def test_judge_refusal_requires_boolean_not_keyword_guess(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({"choices": [{"message": {"content": json.dumps({
                    "correctness": 2, "faithfulness": 2, "abstained": "true",
                    "citation_assessments": []})}}]}).encode()
        with patch("domainrag.answer_evaluation.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "abstained"):
                DeepSeekAnswerJudge(api_key="fake").judge(self.unknown, "无法确定但猜测X", [])


class FrozenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Independent unit fixture: no production corpus or archived benchmark files.
        self.docs = self.root/"data/unit_fixture/docs"
        self.docs.mkdir(parents=True)
        (self.root/"configs").mkdir()
        self.config = self.root/"configs/unit_fixture.json"
        documents = {"cache.md": "Synthetic cache TTL is 5 seconds.",
                     "timeout.md": "Synthetic timeout is 7 seconds."}
        cases = []
        for index, (source, text) in enumerate(documents.items(), 1):
            (self.docs/source).write_text(text, encoding="utf-8")
            cases.append({
                "id": f"fixture-{index:03d}", "question": f"What is synthetic setting {index}?",
                "relevant_sources": [source], "expected_answer": text, "answerable": True,
                "metadata": {"split": "test", "question_type": "direct",
                             "evidence": [{"source": source, "quote": text}]},
            })
        self.case_count = len(cases)
        (self.root/"data/unit_fixture/eval.jsonl").write_text(
            "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
        (self.root/"data/unit_fixture/PROTOCOL.md").write_text(
            "Synthetic unit-test protocol; no empirical scores or frozen research data.", encoding="utf-8")
        (self.root/"data/development_fixture.jsonl").write_text(json.dumps({
            "id": "dev-only", "question": "An unrelated development question?",
            "relevant_sources": ["development.md"], "expected_answer": "development",
        }) + "\n", encoding="utf-8")
        config = {"retriever": "bm25", "docs_path": "data/unit_fixture/docs",
                  "eval_path": "data/unit_fixture/eval.jsonl", "chunk_size": 100,
                  "chunk_overlap": 10, "top_k": 3,
                  "freeze_manifest": "data/unit_fixture/manifest.json"}
        self.config.write_text(json.dumps(config), encoding="utf-8")
        inventory = [self.config, *self.docs.glob("*.md"),
                     self.root/"data/unit_fixture/eval.jsonl", self.root/"data/unit_fixture/PROTOCOL.md"]
        manifest = {
            "dataset_id": "unit_fixture", "frozen_at": "synthetic-test-only",
            "config_paths": ["configs/unit_fixture.json"],
            "docs_path": config["docs_path"], "eval_path": config["eval_path"],
            "protocol_path": "data/unit_fixture/PROTOCOL.md",
            "development_eval_path": "data/development_fixture.jsonl",
            "files": {p.relative_to(self.root).as_posix(): text_sha256(p) for p in inventory},
            "generator_prompt_sha256": hashlib.sha256(
                (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE).encode("utf-8")).hexdigest(),
            "review_status": "synthetic_unit_test_only",
        }
        (self.root/"data/unit_fixture/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_manifest_validates_and_detects_edit(self):
        self.assertEqual(verify_frozen_dataset(self.config)["dataset_id"], "unit_fixture")
        path = self.docs/"cache.md"
        path.write_text(path.read_text(encoding="utf-8") + "changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content changed"):
            verify_frozen_dataset(self.config)

    def test_added_document_and_config_tuning_rejected(self):
        path = self.docs/"new.md"
        path.write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            verify_frozen_dataset(self.config)
        path.unlink()
        cfg = json.loads(self.config.read_text(encoding="utf-8"))
        cfg["top_k"] = 10
        self.config.write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content changed"):
            verify_frozen_dataset(self.config)

    def test_cli_blocks_tampering_before_model_construction(self):
        from domainrag.cli import build_retriever
        with (self.root/"data/unit_fixture/eval.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n")
        with patch("domainrag.cli.build_bm25") as build:
            with self.assertRaises(ValueError): build_retriever(self.config)
            build.assert_not_called()

    def test_existing_report_is_not_overwritten(self):
        from domainrag.holdout import run
        output = self.root/"existing.json"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError): run(self.config, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_failed_run_records_incomplete_not_full_score(self):
        from domainrag.holdout import run
        output = self.root/"failed.json"
        with patch("domainrag.holdout.build_retriever", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError): run(self.config, output, answers=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["answer_evaluation_status"], "incomplete")
        self.assertEqual(len(report["unevaluated_case_ids"]), self.case_count)
        self.assertNotIn("answers", report)

    def test_failure_after_one_case_preserves_checkpoint(self):
        from domainrag.holdout import run
        from domainrag.answer_evaluation import summarize_answers
        output = self.root/"partial.json"
        cfg = json.loads(self.config.read_text(encoding="utf-8"))
        row = evaluate_answers(StaticRetriever(), [EvaluationCase("fixture-001", "?", ("a.md",), "alpha")],
                               Generator(), Judge({"fixture-001": False}), 3)["cases"][0]

        def interrupted(*args, **kwargs):
            kwargs["on_case"](row)
            raise RuntimeError("interrupted after completed case")

        from types import SimpleNamespace
        model = SimpleNamespace(model="fixture", max_tokens=10)
        with patch("domainrag.holdout.build_retriever", return_value=(StaticRetriever(), cfg)), \
             patch("domainrag.holdout.DeepSeekGenerator", return_value=model), \
             patch("domainrag.holdout.DeepSeekAnswerJudge", return_value=model), \
             patch("domainrag.holdout.evaluate_answers", side_effect=interrupted):
            with self.assertRaises(RuntimeError): run(self.config, output, answers=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["answers"], summarize_answers([row]))
        self.assertEqual(len(report["unevaluated_case_ids"]), self.case_count - 1)
        self.assertEqual(report["answer_evaluation_status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
