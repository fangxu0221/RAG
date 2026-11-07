import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from domainrag.agent import AgentLimits, DeepSeekToolModel, run_agent
from domainrag.bm25 import BM25Retriever
from domainrag.chunking import chunk_documents
from domainrag.models import Document
from domainrag.tools import KnowledgeTools


def reply(content=None, calls=None, usage=True, reason=None):
    result = {"choices": [{"message": {"role": "assistant", "content": content,
                                      "tool_calls": calls or []},
                           "finish_reason": reason or ("tool_calls" if calls else "stop")}]}
    if usage:
        result["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return result


def call(identifier="c1", name="search_knowledge", args=None):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args or {"query": "Attention", "top_k": 1})}}


class ScriptedModel:
    model = "offline-test-double"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((json.loads(json.dumps(messages)), kwargs))
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return value


class AgentTests(unittest.TestCase):
    def setUp(self):
        docs = [Document("attention.md", "Attention uses Query Key Value.")]
        self.tools = KnowledgeTools(BM25Retriever(chunk_documents(docs, 100, 0)), docs, {})
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_case(self, replies, **limits):
        model = ScriptedModel(replies)
        result = run_agent("Explain Attention", self.tools, model, Path(self.tmp.name), AgentLimits(**limits))
        rows = [json.loads(x) for x in Path(result["trace_path"]).read_text(encoding="utf-8").splitlines()]
        return result, model, rows

    def test_search_read_answer_and_trace(self):
        result, model, rows = self.run_case([
            reply(calls=[call()]),
            reply(calls=[call("c2", "read_document", {"source": "attention.md"})]),
            reply("Attention uses Query Key Value.[1]")])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["token_usage"]["total_tokens"], 45)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(model.requests[1][0][-1]["tool_call_id"], "c1")
        self.assertEqual(model.requests[2][0][-1]["tool_call_id"], "c2")
        self.assertEqual(rows[-1]["answer"], result["answer"])
        self.assertEqual(rows[-1]["stop_reason"], "model_answered")

    def test_batch_results_pair_with_ids(self):
        result, model, _ = self.run_case([reply(calls=[call(), call("c2", "read_document", {"source": "attention.md"})]), reply("Answer[1]")])
        self.assertEqual(result["tool_attempts"], 2)
        self.assertEqual([m["tool_call_id"] for m in model.requests[1][0] if m["role"] == "tool"], ["c1", "c2"])

    def test_invalid_json_is_returned_to_model(self):
        broken = call()
        broken["function"]["arguments"] = "{broken"
        result, model, _ = self.run_case([reply(calls=[broken]), reply(calls=[call("c2")]), reply("Answer[1]")])
        error = json.loads(model.requests[1][0][-1]["content"])
        self.assertEqual(error["error"]["code"], "invalid_arguments")
        self.assertEqual(result["status"], "completed")

    def test_duplicate_normalizes_defaults(self):
        result, model, rows = self.run_case([
            reply(calls=[call(args={"query": "Attention"})]),
            reply(calls=[call("c2", args={"query": " Attention ", "top_k": 3})]),
            reply("Answer[1]")])
        self.assertEqual(sum(row["event"] == "tool_started" for row in rows), 1)
        self.assertEqual(json.loads(model.requests[2][0][-1]["content"])["error"]["code"], "duplicate_call")

    def test_budget_disables_tools_then_accepts_final(self):
        result, model, _ = self.run_case([reply(calls=[call()]), reply("Answer[1]")], max_tool_calls=1)
        self.assertFalse(model.requests[1][1]["allow_tools"])
        self.assertEqual(result["status"], "completed")

    def test_tool_budget_cannot_be_bypassed(self):
        result, _, rows = self.run_case([reply(calls=[call(), call("c2")])], max_tool_calls=1)
        self.assertEqual(result["stop_reason"], "tool_budget")
        self.assertFalse(any(row["event"] == "tool_started" for row in rows))

    def test_model_and_context_budgets(self):
        result, _, _ = self.run_case([reply(calls=[call()])], max_model_calls=1)
        self.assertEqual(result["stop_reason"], "model_budget")
        result, model, _ = self.run_case([], max_context_chars=1)
        self.assertEqual(result["stop_reason"], "context_budget")
        self.assertEqual(len(model.requests), 0)

    def test_timeout_or_backend_error_does_not_leak(self):
        result, _, rows = self.run_case([TimeoutError("secret-key")])
        self.assertEqual(result["stop_reason"], "model_error")
        self.assertIsNone(result["token_usage"])
        self.assertNotIn("secret-key", json.dumps(rows))

    def test_citation_guards_and_refusal(self):
        for answer, reason in [("Answer[99]", "invalid_citations"), ("Answer", "missing_citations")]:
            result, _, _ = self.run_case([reply(calls=[call()]), reply(answer)])
            self.assertEqual(result["stop_reason"], reason)
            self.assertIsNone(result["answer"])
        result, _, _ = self.run_case([reply("根据当前资料无法确定该答案。")])
        self.assertEqual(result["status"], "abstained")

    def test_unknown_usage_is_not_zero(self):
        result, _, _ = self.run_case([reply(calls=[call()], usage=False), reply("Answer[1]")])
        self.assertIsNone(result["token_usage"])
        self.assertEqual(result["known_token_usage"]["total_tokens"], 15)

    def test_truncated_answer_and_reused_id_rejected(self):
        result, _, _ = self.run_case([reply("partial", reason="length")])
        self.assertEqual(result["stop_reason"], "invalid_model_response")
        result, _, _ = self.run_case([reply(calls=[call()]), reply(calls=[call()])])
        self.assertEqual(result["stop_reason"], "duplicate_call_id")

    def test_transport_payload(self):
        with patch("domainrag.agent.urlopen") as open_url:
            open_url.return_value.__enter__.return_value.read.return_value = json.dumps(reply("ok")).encode()
            model = DeepSeekToolModel(api_key="test-key")
            model.complete([{"role": "user", "content": "test"}], allow_tools=False, timeout=2)
            request = open_url.call_args.args[0]
            body = json.loads(request.data)
            self.assertEqual(body["tool_choice"], "none")
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(len(body["tools"]), 2)
            self.assertNotIn("test-key", request.data.decode())

    def test_limits_reject_bad_values(self):
        for limits in [{"max_tool_calls": True}, {"max_model_calls": 0},
                       {"max_seconds": float("nan")}, {"max_context_chars": -1}]:
            with self.assertRaises(ValueError):
                AgentLimits(**limits)


if __name__ == "__main__":
    unittest.main()
