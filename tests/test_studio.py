"""本地 Studio 的资料、检索、三种问答入口和 Streamlit 交互回归。"""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from domainrag.domain.protocols import ApiCallStats, GenerationResult
from domainrag.io import load_config
from domainrag.models import Document
from test_agent import ScriptedModel, call, reply
from ui import runtime
from ui.catalog import filter_units, load_catalog, question_for


QUESTION = "哪些学生可以购买学生优惠票？"
APP = runtime.ROOT / "ui" / "app.py"


class StudioRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config, self.docs = runtime.snapshot("bm25", {}, 5)
        self.retriever = runtime.prepare_retriever(tuple(self.docs), self.config)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trace_dir = Path(directory.name)

    def test_strategy_presets_and_session_overrides(self):
        for strategy, retriever, rerank in (("bm25", "bm25", False), ("embedding", "embedding", False),
                                           ("rrf", "hybrid", False), ("reranker", "hybrid", True)):
            with self.subTest(strategy=strategy):
                config, documents = runtime.snapshot(strategy, {}, None)
                self.assertEqual(config, load_config(runtime.config_path(strategy)))
                self.assertEqual((config["retriever"], config.get("reranker", {}).get("enabled", False)),
                                 (retriever, rerank))
                self.assertEqual((config["top_k"], config["docs_path"], documents), (3, "data/railway", self.docs))
        config, _ = runtime.snapshot("reranker", {"hybrid": {"candidate_k": 8}, "reranker": {"batch_size": 2}}, 4)
        self.assertEqual((config["hybrid"]["candidate_k"], config["reranker"]["batch_size"], config["top_k"]), (8, 2, 4))
        self.assertEqual(config["reranker"]["backend"], "onnx")
        self.assertEqual(runtime.snapshot("reranker", {}, None)[0]["hybrid"]["candidate_k"], 10)

    def test_catalog_matches_snapshot_and_degrades_when_index_is_missing(self):
        entries = load_catalog(runtime.ROOT / self.config["docs_path"], self.docs)
        self.assertEqual((len(entries), sum(len(entry.units) for entry in entries)), (9, 197))
        self.assertEqual({key: sum(e.category == key for e in entries)
                          for key in ("faq", "notices", "regulations")},
                         {"faq": 6, "notices": 2, "regulations": 1})
        student = next(entry for entry in entries if entry.title == "学生票")
        unit = filter_units(student, "哪些学生")[0]
        self.assertEqual(question_for(student, unit), QUESTION)
        self.assertIn("全日制高等学校", unit.text)
        rules = next(entry for entry in entries if entry.category == "regulations")
        self.assertEqual(len(rules.units), 115)
        self.assertTrue(any("退票" in item.section for item in filter_units(rules, "退票")))

        altered = Document(student.document.source, student.document.text.replace("全日制高等学校", "普通学校"))
        current = load_catalog(runtime.ROOT / self.config["docs_path"], [altered])
        self.assertEqual(len(current), 1)
        self.assertFalse(filter_units(current[0], "哪些学生"))
        fallback = load_catalog(self.trace_dir, [Document("notices/sample.md", "# 中文标题\n正文")])[0]
        self.assertEqual((fallback.title, fallback.category, fallback.units), ("中文标题", "notices", ()))

    def test_all_modes_reuse_retriever_and_keep_actual_tool_top_k(self):
        generator = Mock()
        generator.generate.return_value = GenerationResult("学生票有资格要求[2]。", ApiCallStats(0.1, 11, 4, 15))
        spy = Mock(wraps=self.retriever)
        modes = ["rag", "agent"]
        if find_spec("langgraph"):
            modes.append("langgraph")
        for mode in modes:
            model = ScriptedModel([
                reply(calls=[call(args={"query": QUESTION, "top_k": 2})]),
                reply(calls=[call("c2", args={"query": QUESTION})]),
                reply(content="学生票有资格要求[2]。"),
            ])
            with self.subTest(mode=mode), patch.object(runtime, "DeepSeekGenerator", return_value=generator), \
                    patch.object(runtime, "DeepSeekToolModel", return_value=model):
                result = runtime.run_answer(mode, QUESTION, self.config, self.docs, spy, self.trace_dir)
                self.assertEqual(result["status"], "completed")
                events = runtime.read_trace(Path(result["trace_path"]))
                reference = result["citations"][0]
                self.assertEqual(reference["citation_id"], 2)
                self.assertEqual(reference["chunk_id"], result["evidence"][1]["chunk_id"])
                self.assertTrue(all("messages" not in event and "message" not in event for event in events))
                if mode == "rag":
                    self.assertEqual(result["token_usage"]["total_tokens"], 15)
                    self.assertEqual(len(result["evidence"]), 5)
                else:
                    args = [event["arguments"] for event in events if event["event"] == "tool_started"]
                    self.assertEqual(args, [{"query": QUESTION, "top_k": 2}, {"query": QUESTION}])
                    results = [event["observation"]["result"] for event in events if event["event"] == "tool_observation"]
                    self.assertEqual([data["count"] for data in results], [2, 3])
        self.assertEqual([args[0][1] for args in spy.search.call_args_list], [5] + [2, 3] * (len(modes) - 1))

    def test_fixed_rag_timeout_refusal_and_invalid_citation(self):
        for response, status, reason in (
            (TimeoutError("sensitive exception text"), "failed", "model_timeout"),
            (GenerationResult("根据当前资料无法确定。"), "abstained", "model_abstained"),
            (GenerationResult("无效引用[99]。"), "failed", "invalid_citations"),
        ):
            generator = Mock()
            if isinstance(response, Exception):
                generator.generate.side_effect = response
            else:
                generator.generate.return_value = response
            result = runtime.run_fixed_rag(QUESTION, self.retriever, self.config, generator, self.trace_dir)
            self.assertEqual((result["status"], result["stop_reason"], result["model_calls"]), (status, reason, 1))
            self.assertTrue(result["evidence"])
            trace = Path(result["trace_path"]).read_text(encoding="utf-8")
            self.assertNotIn("sensitive exception text", trace)
            self.assertEqual(json.loads(trace.splitlines()[-1])["stop_reason"], reason)


@unittest.skipUnless(find_spec("streamlit"), "requires the UI extra")
class StudioAppTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest

        env = patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        self.app = AppTest.from_file(str(APP), default_timeout=15).run()

    def page(self, name):
        self.app.radio(key="page").set_value(name).run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(item for item in self.app.button if item.label == label)

    def apply_bm25(self, top_k=4):
        self.page("检索设置")
        next(item for item in self.app.radio if item.label == "检索策略").set_value("bm25").run()
        next(item for item in self.app.number_input if item.label == "Top-K").set_value(top_k)
        self.button("应用检索设置").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["strategy"], "bm25")

    def test_browse_filter_open_and_transfer_question(self):
        self.page("知识库浏览")
        self.assertIn("9 份检索资料 · 197 个知识单元", self.app.caption[0].value)
        self.assertEqual(self.app.radio[0].options, ["常见问题", "使用须知", "相关规程"])
        self.app.radio[0].set_value("regulations").run()
        self.assertEqual(self.app.selectbox[1].label, "按章和节查看")
        self.app.text_input[1].set_value("退票").run()
        self.assertTrue(any("退票" in option for option in self.app.selectbox[1].options))
        self.app.radio[0].set_value("faq").run()
        self.app.text_input[0].set_value("学生票")
        self.app.text_input[1].set_value("哪些学生").run()
        self.assertEqual(self.app.selectbox[0].options, ["学生票 · 17 条"])
        self.assertEqual(self.app.selectbox[1].options, [f"2. {QUESTION}"])
        self.assertTrue(any("全日制高等学校" in row.value for row in self.app.markdown))
        self.button("用此问题测试").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio(key="page").value, "问答测试")
        self.assertEqual(self.app.text_area[0].value, QUESTION)

    def test_settings_draft_and_offline_debug(self):
        self.app.text_area[0].set_value(QUESTION).run()
        self.apply_bm25()
        self.page("检索调试")
        self.app.text_input[0].set_value(QUESTION)
        self.button("开始检索").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.expander), 4)
        self.assertTrue(any("学生票" in exp.label for exp in self.app.expander))
        self.assertTrue(any("原始相关性分数" in row.value for row in self.app.caption))
        self.page("问答测试")
        self.assertEqual(self.app.text_area[0].value, QUESTION)
        self.assertEqual(self.app.session_state["top_k"], 4)
        self.assertTrue(self.button("发送").disabled)
        self.assertTrue(any("DEEPSEEK_API_KEY" in row.value for row in self.app.warning))

        self.page("检索调试")
        self.app.selectbox(key="_debug_strategy").set_value("rrf").run()
        self.assertFalse(self.app.exception)
        self.page("问答测试")
        self.assertEqual(self.app.session_state["strategy"], "rrf")
        self.assertEqual(self.app.session_state["top_k"], 4)

    def test_retriever_cache_tracks_documents_and_index_settings(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_string("""
import streamlit as st
from domainrag.models import Document
from ui import runtime
from ui.app import get_retriever

config, docs = runtime.snapshot("bm25", {}, None)
retriever = get_retriever(config, docs)
changed_config = {**config, "bm25": {"k1": 1.0, "b": 0.75}}
changed_docs = [Document(docs[0].source, docs[0].text + "新增内容"), *docs[1:]]
st.session_state["same_with_top_k"] = retriever is get_retriever({**config, "top_k": 6}, docs)
st.session_state["same_with_new_config"] = retriever is get_retriever(changed_config, docs)
st.session_state["same_with_new_docs"] = retriever is get_retriever(config, changed_docs)
""", default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["same_with_top_k"])
        self.assertFalse(app.session_state["same_with_new_config"])
        self.assertFalse(app.session_state["same_with_new_docs"])

    def test_submit_is_explicit_and_rerun_does_not_repeat_model_call(self):
        self.apply_bm25()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-test-key"}), \
                patch.object(runtime, "TRACE_DIR", Path(directory)), \
                patch.object(runtime, "DeepSeekGenerator") as factory:
            generator = factory.return_value
            generator.generate.return_value = GenerationResult("学生优惠票有资格要求[1]。")
            self.page("问答测试")
            self.app.text_area[0].set_value(QUESTION).run()
            generator.generate.assert_not_called()
            self.button("发送").click().run()
            self.assertFalse(self.app.exception)
            self.app.run()
            generator.generate.assert_called_once()
            self.assertTrue(any("引用证据" in row.value for row in self.app.markdown))
            self.assertTrue(any("成功" in row.value for row in self.app.caption))

    def test_agent_trace_ui_shows_tool_top_k_and_usage(self):
        self.apply_bm25(top_k=5)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-test-key"}), \
                patch.object(runtime, "TRACE_DIR", Path(directory)), \
                patch.object(runtime, "DeepSeekToolModel") as factory:
            for mode in ("agent", "langgraph") if find_spec("langgraph") else ("agent",):
                with self.subTest(mode=mode):
                    answer = reply(content="学生优惠票有资格要求[1]。")
                    answer["choices"][0]["message"]["reasoning_content"] = "private-model-data"
                    factory.return_value = ScriptedModel([
                        reply(calls=[call(args={"query": QUESTION, "top_k": 2})]), answer,
                    ])
                    self.page("问答测试")
                    self.app.radio(key="_mode").set_value(mode)
                    self.app.text_area[0].set_value(QUESTION).run()
                    self.button("发送").click().run()
                    self.assertFalse(self.app.exception)
                    self.assertTrue(any("请求 Top-K 2" in row.value for row in self.app.text))
                    self.assertTrue(any("实际返回 2 条" in row.value for row in self.app.caption))
                    self.assertTrue(any("合计 30" in row.value for row in self.app.text))
            for path in Path(directory).glob("*.jsonl"):
                trace = path.read_text(encoding="utf-8")
                self.assertNotIn("private-model-data", trace)
                self.assertNotIn("offline-test-key", trace)

    def test_optional_dependencies_have_install_messages(self):
        with patch.object(runtime, "find_spec", return_value=None):
            self.app.radio(key="_mode").set_value("langgraph").run()
            self.assertFalse(self.app.exception)
            messages = [row.value for row in self.app.warning]
            self.assertTrue(any("[langgraph]" in message for message in messages))
            self.assertTrue(any("[reranker]" in message for message in messages))
            self.assertTrue(self.button("发送").disabled)
