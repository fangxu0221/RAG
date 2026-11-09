import unittest

from domainrag.evidence import EvidenceGroundTruth
from domainrag.models import Chunk, Document, EvaluationCase, SearchResult
from scripts import railpolicy_v2_auto as v2


class RailwayV2AutoTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"id": "x", "required_points": [0, 1], "location_valid": True}

    def run_vote(self, model, status, points):
        return {"requested_model": model, "status": "completed", "decisions": {
            "x": {"status": status, "supported_points": points, "reason": "Fixture judgment"}}}

    def test_two_models_must_agree_on_every_required_point(self):
        flash = self.run_vote("flash", "accepted", [0, 1])
        pro = self.run_vote("pro", "accepted", [0, 1])
        self.assertEqual(v2.adjudicate(self.candidate, [flash, pro])["status"], "accepted")
        partial = self.run_vote("pro", "accepted", [0])
        reject = self.run_vote("pro", "rejected", [0])
        missing = {"requested_model": "pro", "decisions": {}, "status": "unavailable"}
        for second in (partial, reject, missing, flash):
            self.assertEqual(v2.adjudicate(self.candidate, [flash, second])["status"], "uncertain")
        invalid = dict(self.candidate, location_valid=False)
        self.assertEqual(v2.adjudicate(invalid, [flash, pro])["status"], "uncertain")
        self.assertEqual(v2.adjudicate(self.candidate, [
            self.run_vote("flash", "rejected", [0]), reject])["status"], "rejected")

    def test_malformed_or_duplicate_decisions_fail_closed(self):
        vote = {"id": "x", "status": "accepted", "supported_points": [0, 1], "reason": "Both points"}
        for decisions in ([vote, vote], [dict(vote, id="unknown")],
                          [dict(vote, supported_points=[True, 1])], [dict(vote, reason="")]):
            with self.assertRaises(ValueError):
                v2.parse_decisions({"decisions": decisions}, {"x"})

    def test_unconfirmed_candidates_stay_out_of_scoring_evidence(self):
        _, _, rows, units, chunks, truth, inputs = v2.load_snapshot()
        candidates = v2.propose(rows, units, chunks, truth)
        audit = {"inputs": inputs, "candidates": candidates, "runs": [
            {"requested_model": model, "status": "unavailable", "decisions": {}} for model in v2.REVIEWERS]}
        revised, records = v2.build_rows(rows, audit)
        self.assertEqual(len(candidates), 26)
        self.assertTrue(all(c["location_valid"] for c in candidates))
        self.assertTrue(all(r["status"] == "uncertain" for r in records.values()))
        for old, new in zip(rows, revised):
            for field in ("id", "question", "expected_answer", "answerable"):
                self.assertEqual(old[field], new[field])
            self.assertEqual(old["metadata"]["answer_key_points"], new["metadata"]["answer_key_points"])
            self.assertEqual(old["relevant_sources"], new["relevant_sources"])
            self.assertEqual(len(old["metadata"]["evidence"]), len(new["metadata"]["evidence"]))
            self.assertFalse(new["metadata"]["v2_auto"]["human_reviewed"])

    def test_source_group_recall_uses_or_within_and_across_groups(self):
        documents = [Document("faq", "A"), Document("rules", "A"), Document("notice", "B")]
        case = EvaluationCase("test", "question", ("faq", "rules", "notice"), metadata={"evidence": [
            {"source": "faq", "quote": "A", "group_id": "first"},
            {"source": "rules", "quote": "A", "group_id": "first"},
            {"source": "notice", "quote": "B", "group_id": "second"},
        ]})
        groups = EvidenceGroundTruth(documents, [case]).groups["test"]
        first = SearchResult(Chunk("rules#0", "rules", "A", 0, 1), 1)
        second = SearchResult(Chunk("notice#0", "notice", "B", 0, 1), 0)
        self.assertEqual(v2.source_groups(groups, [first])["source_group_recall@3"], 0.5)
        self.assertEqual(v2.source_groups(groups, [first, second])["source_group_recall@3"], 1.0)
        self.assertTrue(v2.source_groups(groups, [first, second])["source_group_sufficient@3"])
        self.assertIsNone(v2.source_groups((), [first])["source_group_recall@3"])

    def test_saved_ranking_rejects_coordinate_or_rank_tampering(self):
        chunks = {f"a#{i}": Chunk(f"a#{i}", "a", str(i), i, i + 1) for i in range(3)}
        rows = [{"rank": i + 1, "chunk_id": c.chunk_id, "source": c.source, "start": c.start,
                 "end": c.end, "preview": c.text, "score": float(i)}
                for i, c in enumerate(chunks.values())]
        self.assertEqual([r.chunk.chunk_id for r in v2.restore_results(rows, chunks)], list(chunks))
        for changed in (dict(rows[0], start=1), dict(rows[0], rank=2), dict(rows[0], preview="wrong")):
            with self.assertRaises(ValueError):
                v2.restore_results([changed, *rows[1:]], chunks)
        with self.assertRaises(ValueError):
            v2.restore_results(rows[:2], chunks)
