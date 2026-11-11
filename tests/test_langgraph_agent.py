"""用相同离线脚本对照 StateGraph 与原生 Agent。"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from domainrag import agent, langgraph_cli
from domainrag.application.knowledge_service import KnowledgeTools, ToolSession
from domainrag.chunking import chunk_documents
from domainrag.infrastructure.retrievers.bm25 import BM25Retriever
from domainrag.models import Document
from domainrag.workflows import langgraph_agent, native_agent

from test_agent import ScriptedModel, call, reply


class ManualClock:
    now = 0.0

    def __call__(self):
        return self.now


def normalize(value):
    """保留业务字段，并统一嵌套在工具消息里的请求/执行 ID。"""
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: normalize(item) for key, item in value.items()
        if key not in {"request_id", "trace_path", "timestamp", "elapsed_ms"}
    }
    if "call_id" in normalized:
        normalized["call_id"] = normalized["call_id"].split(":")[-1]
    if value.get("role") == "tool":
        normalized["content"] = normalize(json.loads(value["content"]))
    return normalized


def tool_messages(request):
    messages, _ = request
    return [message for message in messages if message["role"] == "tool"]


@unittest.skipUnless(find_spec("langgraph"), "install the langgraph extra")
class LangGraphAgentTests(unittest.TestCase):
    def setUp(self):
        docs = [Document("attention.md", "Attention uses Query Key Value.")]
        self.tools = KnowledgeTools(BM25Retriever(chunk_documents(docs, 100, 0)), docs, {})
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trace_dir = Path(directory.name)

    def run_case(self, runner, replies, *, model_seconds=0, search_seconds=0,
                 search_error=None, **limits):
        clock = ManualClock()
        model = ScriptedModel(deepcopy(replies))
        scripted_complete = model.complete
        bm25_search = self.tools.search_knowledge

        def complete(messages, **kwargs):
            response = scripted_complete(messages, **kwargs)
            clock.now += model_seconds
            return response

        def search(*args, **kwargs):
            clock.now += search_seconds
            if search_error is not None:
                raise search_error
            return bm25_search(*args, **kwargs)

        with (patch.object(native_agent, "perf_counter", clock),
              patch.object(langgraph_agent, "perf_counter", clock),
              patch.object(model, "complete", side_effect=complete),
              patch.object(self.tools, "search_knowledge", side_effect=search)):
            result = runner(" Explain Attention ", self.tools, model, self.trace_dir,
                            native_agent.AgentLimits(**limits))
        rows = [json.loads(line) for line in
                Path(result["trace_path"]).read_text(encoding="utf-8").splitlines()]
        return result, model.requests, rows

    def assert_parity(self, replies, **kwargs):
        native = self.run_case(native_agent.run_agent, replies, **kwargs)
        graph = self.run_case(langgraph_agent.run_langgraph_agent, replies, **kwargs)
        self.assertEqual(normalize(graph[0]), normalize(native[0]))
        self.assertEqual([(normalize(messages), options) for messages, options in graph[1]],
                         [(normalize(messages), options) for messages, options in native[1]])
        self.assertEqual(normalize(graph[2]), normalize(native[2]))
        return graph

    def test_search_read_cite_and_deduplicate_evidence(self):
        result, requests, rows = self.assert_parity([
            reply(calls=[call()]),
            reply(calls=[call("read", "read_document", {"source": "attention.md"})]),
            reply("Attention uses Query Key Value.[1]"),
        ])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["citations"], result["evidence"])
        self.assertEqual([item["citation_id"] for item in result["evidence"]], [1])
        self.assertEqual(result["token_usage"]["total_tokens"], 45)
        self.assertEqual([message["tool_call_id"] for message in tool_messages(requests[2])],
                         ["c1", "read"])
        self.assertEqual(rows[-1]["tool_calls"], 2)

    def test_batch_order_and_ids(self):
        result, requests, rows = self.assert_parity([
            reply(calls=[call(), call("read", "read_document",
                                      {"source": "attention.md", "max_chars": 9})]),
            reply("Answer[2][1][2]"),
        ])
        self.assertEqual(result["tool_attempts"], 2)
        self.assertEqual([item["citation_id"] for item in result["citations"]], [1, 2])
        self.assertEqual([message["tool_call_id"] for message in tool_messages(requests[1])],
                         ["c1", "read"])
        self.assertEqual([row["tool"] for row in rows if row["event"] == "tool_started"],
                         ["search_knowledge", "read_document"])

    def test_invalid_arguments_then_recovery(self):
        for invalid_json in ("{broken", "[]", '{"query":NaN}'):
            with self.subTest(arguments=invalid_json):
                broken = call()
                broken["function"]["arguments"] = invalid_json
                result, requests, rows = self.assert_parity([
                    reply(calls=[broken]), reply(calls=[call("fixed")]), reply("Answer[1]"),
                ])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["tool_attempts"], 2)
                self.assertEqual(rows[-1]["tool_calls"], 1)
                observation = json.loads(tool_messages(requests[1])[0]["content"])
                self.assertEqual(observation["error"]["code"], "invalid_arguments")

    def test_defaults_and_duplicate_calls_count_as_attempts(self):
        for name, args, explicit in (
            ("search_knowledge", {"query": "Attention"}, {"query": " Attention ", "top_k": 3}),
            ("read_document", {"source": "attention.md"},
             {"source": "attention.md", "start": 0, "max_chars": 2000}),
        ):
            with self.subTest(tool=name):
                result, requests, rows = self.assert_parity([
                    reply(calls=[call("c1", name, args)]),
                    reply(calls=[call("c2", name, explicit)]),
                    reply("Answer[1]"),
                ], max_tool_calls=2)
                self.assertEqual(result["tool_attempts"], 2)
                self.assertEqual(rows[-1]["tool_calls"], 1)
                observation = json.loads(tool_messages(requests[2])[-1]["content"])
                self.assertEqual(observation["error"]["code"], "duplicate_call")
                self.assertFalse(requests[2][1]["allow_tools"])

    def test_duplicate_provider_ids_within_and_across_batches(self):
        for replies, reason, tool_calls in (
            ([reply(calls=[call(), call()])], "model_error", 0),
            ([reply(calls=[call()]), reply(calls=[call()])], "duplicate_call_id", 1),
        ):
            with self.subTest(reason=reason):
                result, _, rows = self.assert_parity(replies)
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(rows[-1]["tool_calls"], tool_calls)

    def test_tool_budget_allows_final_answer_but_rejects_further_calls(self):
        for final, reason in (
            (reply("Answer[1]"), "model_answered"),
            (reply(calls=[call("extra")]), "tool_budget"),
        ):
            with self.subTest(reason=reason):
                result, requests, rows = self.assert_parity([reply(calls=[call()]), final],
                                                            max_tool_calls=1)
                self.assertFalse(requests[1][1]["allow_tools"])
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(rows[-1]["tool_calls"], 1)

    def test_over_budget_batch_does_not_execute_prefix(self):
        result, _, rows = self.assert_parity([reply(calls=[call(), call("c2")])], max_tool_calls=1)
        self.assertEqual(result["stop_reason"], "tool_budget")
        self.assertEqual(result["tool_attempts"], 0)
        self.assertEqual(rows[-1]["tool_calls"], 0)

    def test_model_budget_reaches_maximum_after_tools(self):
        replies = [reply(calls=[call(f"c{i}", args={"query": f"Attention {i}"})])
                   for i in range(20)]
        result, requests, rows = self.assert_parity(replies, max_model_calls=20, max_tool_calls=20)
        self.assertEqual(result["stop_reason"], "model_budget")
        self.assertEqual(result["model_calls"], 20)
        self.assertEqual(len(requests), 20)
        self.assertEqual(rows[-1]["tool_calls"], 20)

    def test_context_budget_before_first_model_or_after_tool(self):
        result, requests, _ = self.assert_parity([], max_context_chars=1)
        self.assertEqual(result["stop_reason"], "context_budget")
        self.assertEqual(requests, [])
        initial_size = len(native_agent._json(native_agent._initial_messages("Explain Attention")))
        result, requests, _ = self.assert_parity([reply(calls=[call()])],
                                                max_context_chars=initial_size)
        self.assertEqual(result["stop_reason"], "context_budget")
        self.assertEqual(len(requests), 1)

    def test_time_budget_after_model(self):
        result, requests, rows = self.assert_parity([reply(calls=[call()])],
                                                     max_seconds=5, model_seconds=5)
        self.assertEqual(result["stop_reason"], "time_budget")
        self.assertEqual(requests[0][1]["timeout"], 5)
        self.assertEqual(result["token_usage"]["total_tokens"], 15)
        self.assertEqual(rows[-1]["tool_calls"], 0)

    def test_time_budget_partway_through_batch_preserves_evidence(self):
        result, _, rows = self.assert_parity([reply(calls=[call(), call("later")])],
                                              max_seconds=5, search_seconds=5)
        self.assertEqual(result["stop_reason"], "time_budget")
        self.assertEqual(result["tool_attempts"], 1)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(rows[-1]["tool_calls"], 1)

    def test_model_timeout_cap_and_remaining_budget(self):
        result, requests, _ = self.assert_parity([reply(calls=[call()]), reply("Answer[1]")],
                                                 model_seconds=1, search_seconds=40)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([options["timeout"] for _, options in requests], [60, 49])

    def test_model_errors_are_sanitized(self):
        for exc in (TimeoutError("secret-sentinel"), RuntimeError("secret-sentinel")):
            with self.subTest(error=type(exc).__name__):
                result, _, rows = self.assert_parity([exc])
                self.assertEqual(result["stop_reason"], "model_error")
                self.assertIsNone(result["token_usage"])
                self.assertFalse(result["usage_complete"])
                self.assertNotIn("secret-sentinel", json.dumps(rows))
                event = next(row for row in rows if row["event"] == "model_failed")
                self.assertEqual(event["error_type"], type(exc).__name__)

    def test_tool_backend_error_becomes_sanitized_observation(self):
        result, requests, rows = self.assert_parity([
            reply(calls=[call()]), reply("根据当前资料无法确定"),
        ], search_error=RuntimeError("secret-sentinel"))
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(rows[-1]["failed_calls"], 1)
        observation = json.loads(tool_messages(requests[1])[0]["content"])
        self.assertEqual(observation["error"]["code"], "tool_failed")
        self.assertNotIn("secret-sentinel", json.dumps(rows))

    def test_invalid_finish_reasons_and_malformed_response(self):
        wrong_arguments_type = call()
        wrong_arguments_type["function"]["arguments"] = {}
        wrong_role = reply("Answer[1]")
        wrong_role["choices"][0]["message"]["role"] = "user"
        for response, reason in (
            (reply("partial", reason="length"), "invalid_model_response"),
            (reply("  "), "invalid_model_response"),
            (reply("looks final", [call()], reason="stop"), "invalid_model_response"),
            (reply("looks final", [wrong_arguments_type]), "model_error"),
            (wrong_role, "model_error"),
            ({}, "model_error"),
        ):
            with self.subTest(response=response):
                result, _, rows = self.assert_parity([response])
                self.assertEqual(result["stop_reason"], reason)
                self.assertIsNone(result["answer"])
                self.assertEqual(rows[-1]["tool_calls"], 0)

    def test_citation_guards_and_refusal(self):
        for replies, status, reason in (
            ([reply(calls=[call()]), reply("Answer[0]")], "failed", "invalid_citations"),
            ([reply(calls=[call()]), reply("Answer[2]")], "failed", "invalid_citations"),
            ([reply(calls=[call()]), reply("Answer")], "failed", "missing_citations"),
            ([reply("Answer")], "failed", "missing_citations"),
            ([reply("根据当前资料无法确定")], "abstained", "model_abstained"),
            ([reply(calls=[call()]), reply("根据当前资料无法确定。[1]")],
             "abstained", "model_abstained"),
        ):
            with self.subTest(status=status, reason=reason, replies=replies):
                result, _, _ = self.assert_parity(replies)
                self.assertEqual((result["status"], result["stop_reason"]), (status, reason))

    def test_unknown_usage_and_known_usage_before_parser_failure(self):
        result, _, _ = self.assert_parity([
            reply(calls=[call()], usage=False), reply("Answer[1]"),
        ])
        self.assertIsNone(result["token_usage"])
        self.assertEqual(result["known_token_usage"]["total_tokens"], 15)
        self.assertFalse(result["usage_complete"])
        malformed = reply("bad")
        malformed["choices"] = []
        result, _, _ = self.assert_parity([malformed])
        self.assertEqual(result["known_token_usage"]["total_tokens"], 15)
        self.assertIsNone(result["token_usage"])

    def test_session_lifecycle_and_consecutive_requests(self):
        sessions = []

        def create_session(*args):
            session = ToolSession(*args)
            sessions.append(session)
            return session

        replies = [reply(calls=[call()]), reply("Answer[1]")]
        with patch.object(langgraph_agent, "ToolSession", side_effect=create_session):
            first, _, first_rows = self.run_case(langgraph_agent.run_langgraph_agent, replies)
            second, _, second_rows = self.run_case(langgraph_agent.run_langgraph_agent, replies)
        self.assertEqual(len(sessions), 2)
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertNotEqual(first["trace_path"], second["trace_path"])
        self.assertEqual(normalize(first), normalize(second))
        for session, rows in zip(sessions, (first_rows, second_rows)):
            self.assertEqual({row["request_id"] for row in rows}, {session.request_id})
            self.assertEqual([row["event"] for row in rows].count("request_started"), 1)
            self.assertEqual([row["event"] for row in rows].count("request_finished"), 1)
            self.assertEqual(rows[-1]["status"], "completed")
            with self.assertRaisesRegex(RuntimeError, "session is closed"):
                session.record({"event": "unexpected"})

    def test_graph_error_aborts_and_closes_session(self):
        with patch.object(langgraph_agent, "_build_graph", side_effect=RuntimeError("graph error")):
            with self.assertRaisesRegex(RuntimeError, "graph error"):
                self.run_case(langgraph_agent.run_langgraph_agent, [])
        trace_path, = self.trace_dir.glob("*.jsonl")
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["event"], "request_finished")
        self.assertEqual(rows[-1]["stop_reason"], "caller_aborted")

    def test_graph_executes_real_model_tool_model_cycle(self):
        from langgraph.graph import StateGraph

        order = []
        add_node = StateGraph.add_node

        def track(builder, name, node):
            def tracked(state):
                order.append(name)
                return node(state)

            return add_node(builder, name, tracked)

        with (patch.object(StateGraph, "add_node", track),
              patch.object(native_agent, "run_agent", side_effect=AssertionError("native loop"))):
            self.run_case(langgraph_agent.run_langgraph_agent,
                          [reply(calls=[call()]), reply("Answer[1]")])
        self.assertEqual(order, ["model", "tools", "model", "validate_answer", "finalize"])

    def test_cli_uses_shared_options_and_json_exit_contract(self):
        config_path = Path("offline-test.json")
        argv = ["langgraph_cli", "--config", str(config_path), "--question", "Explain Attention",
                "--model", "offline-model", "--trace-dir", str(self.trace_dir),
                "--max-tool-calls", "1", "--max-model-calls", "2", "--max-seconds", "30"]
        for replies, expected_status, exit_code in (
            ([reply(calls=[call()]), reply("Answer[1]")], "completed", 0),
            ([reply("根据当前资料无法确定")], "abstained", 0),
            ([reply(calls=[call(), call("extra")])], "stopped", 1),
            ([reply("Answer")], "failed", 1),
        ):
            with self.subTest(status=expected_status):
                model = ScriptedModel(replies)
                output = StringIO()
                with (patch.object(sys, "argv", argv), redirect_stdout(output),
                      patch.object(agent, "DeepSeekToolModel", return_value=model) as model_factory,
                      patch.object(KnowledgeTools, "from_config", return_value=self.tools) as tools_factory,
                      self.assertRaises(SystemExit) as exit_error):
                    langgraph_cli.main()
                model_factory.assert_called_once_with("offline-model")
                tools_factory.assert_called_once_with(config_path)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(exit_error.exception.code, exit_code)
                self.assertEqual(Path(result["trace_path"]).parent, self.trace_dir)


class OptionalDependencyTests(unittest.TestCase):
    def test_cli_reports_missing_extra_before_assembling_dependencies(self):
        argv = ["langgraph_cli", "--config", "offline-test.json", "--question", "Attention"]
        error_output = StringIO()
        with (patch.object(sys, "argv", argv), redirect_stderr(error_output),
              patch.object(langgraph_cli, "find_spec", return_value=None),
              patch.object(agent, "DeepSeekToolModel") as model_factory,
              patch.object(KnowledgeTools, "from_config") as tools_factory,
              self.assertRaises(SystemExit) as exit_error):
            langgraph_cli.main()
        self.assertEqual(exit_error.exception.code, 2)
        self.assertIn('python -m pip install -e ".[langgraph]"', error_output.getvalue())
        model_factory.assert_not_called()
        tools_factory.assert_not_called()
