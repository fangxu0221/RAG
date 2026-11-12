"""从项目根目录执行 python -m streamlit run ui/app.py。"""

from pathlib import Path
from time import perf_counter

import streamlit as st

from domainrag.bootstrap import project_path
from domainrag.io import load_config
from ui.catalog import CATEGORIES, filter_units, load_catalog, question_for
from ui import runtime


QA, LIBRARY, DEBUG, SETTINGS = "问答测试", "知识库浏览", "检索调试", "检索设置"
STATUS = {"completed": "成功", "abstained": "拒答", "failed": "失败", "stopped": "预算停止"}
STOP_REASONS = {
    "model_answered": "模型已作答", "model_abstained": "当前资料不足",
    "model_error": "模型调用失败", "model_timeout": "模型请求超时",
    "retrieval_failed": "检索失败，请检查模型依赖和权重",
    "invalid_model_response": "模型响应格式无效", "invalid_citations": "引用编号超出证据范围",
    "missing_citations": "回答缺少引用", "model_budget": "达到模型调用次数上限",
    "tool_budget": "达到工具调用次数上限", "time_budget": "达到运行时间上限",
    "context_budget": "达到上下文长度上限", "duplicate_call_id": "模型重复使用工具调用编号",
}
EVENT_NAMES = {
    "retrieval_started": "开始检索", "retrieval_finished": "检索完成",
    "model_started": "开始模型调用", "model_finished": "模型返回", "model_failed": "模型调用失败",
    "tool_started": "开始执行工具", "tool_finished": "工具执行结束", "tool_observation": "工具返回给模型",
}
TOOL_ERRORS = {"invalid_arguments": "参数无效", "tool_failed": "工具执行失败",
               "duplicate_call": "重复的工具请求"}

cached_embedding = st.cache_resource(show_spinner=False, max_entries=4)(runtime.prepare_embedding)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_retriever(documents, index_config):
    return runtime.prepare_retriever(
        documents, index_config,
        embedding_builder=lambda chunks, config: cached_embedding(tuple(chunks), config.get("embedding", {})),
    )


def navigate(page, question=None):
    st.session_state.page = page
    if question is not None:
        st.session_state.question = question
        st.session_state.pop("_question", None)


def remember_question():
    st.session_state.question = st.session_state._question


def render_catalog(entries):
    st.title(LIBRARY)
    st.caption(f"{len(entries)} 份检索资料 · {sum(len(entry.units) for entry in entries)} 个知识单元。"
               "知识单元按原文结构组织，检索时另行切分为 Chunk。")
    categories = [category for category in CATEGORIES if any(entry.category == category for entry in entries)]
    category = st.radio("资料分类", categories, format_func=CATEGORIES.__getitem__, horizontal=True)
    visible = [entry for entry in entries if entry.category == category]
    selector, search = st.columns([3, 2])
    with search:
        keyword = st.text_input("搜索资料标题", placeholder="例如：学生票").strip().casefold()
    visible = [entry for entry in visible if keyword in entry.title.casefold()]
    if not visible:
        st.info("当前分类没有匹配的资料。")
        return
    with selector:
        entry = st.selectbox("选择资料", visible, format_func=lambda item: (
            f"{item.title} · {len(item.units)} 条" if item.units else item.title
        ))
    st.subheader(entry.title)
    if entry.url:
        st.link_button("查看 12306 官方来源", entry.url)
    if not entry.units:
        st.info("当前快照没有可用的知识条目索引，可以展开阅读整份资料。")
        with st.expander("查看资料原文"):
            st.markdown(entry.document.text)
        return
    units = filter_units(entry, st.text_input("搜索知识条目", placeholder="输入中文关键词"))
    if not units:
        st.info("没有匹配的知识条目。")
        return
    if entry.category == "regulations":
        groups = {}
        for unit in units:
            label = " · ".join(filter(None, (unit.chapter, unit.section))) or "其他条款"
            groups.setdefault(label, []).append(unit)
        group = st.selectbox("按章和节查看", list(groups))
        units = groups[group]
    unit = st.selectbox(f"选择知识条目（{len(units)} 条）", units, format_func=lambda item: item.title)
    with st.expander("查看条目原文", expanded=True):
        st.markdown(unit.text)
    st.button("用此问题测试", on_click=navigate, args=(QA, question_for(entry, unit)), type="primary")


def render_settings():
    st.title(SETTINGS)
    state = st.session_state
    strategy = st.radio("检索策略", list(runtime.STRATEGIES),
                        index=list(runtime.STRATEGIES).index(state.strategy),
                        format_func=lambda key: runtime.STRATEGIES[key].label)
    try:
        config = load_config(runtime.config_path(strategy))
    except (OSError, ValueError):
        st.error("无法读取所选策略的铁路配置，请检查 configs 目录。")
        return
    saved = state.overrides.get(strategy, {})
    overrides = {}
    with st.form(f"settings_{strategy}"):
        top_k = st.number_input("Top-K", min_value=1, max_value=10, step=1,
                                value=state.top_k or config["top_k"],
                                help="纯 RAG 和检索调试使用此值。Agent 可指定 Top-K，省略时沿用工具默认值 3。")
        with st.expander("高级选项"):
            if "bm25" in config:
                bm25 = {**config["bm25"], **saved.get("bm25", {})}
                first, second = st.columns(2)
                overrides["bm25"] = {
                    "k1": first.number_input("BM25 · k1", min_value=0.01, value=float(bm25["k1"]), step=0.05),
                    "b": second.number_input("BM25 · b", min_value=0.0, max_value=1.0,
                                             value=float(bm25["b"]), step=0.05),
                }
            if "embedding" in config:
                st.caption(f"BGE 模型 · {config['embedding']['model']}")
            if "hybrid" in config:
                hybrid = {**config["hybrid"], **saved.get("hybrid", {})}
                first, second = st.columns(2)
                overrides["hybrid"] = {
                    "candidate_k": first.number_input("RRF · 每路候选数", min_value=1, value=hybrid["candidate_k"]),
                    "rrf_k": second.number_input("RRF · 排名常数", min_value=0, value=hybrid["rrf_k"]),
                }
            if config.get("reranker", {}).get("enabled"):
                reranker = {**config["reranker"], **saved.get("reranker", {})}
                first, second = st.columns(2)
                overrides["reranker"] = {
                    "model": st.text_input("重排模型", value=reranker["model"]).strip(),
                    "candidate_k": first.number_input("重排 · 候选数", min_value=1, value=reranker["candidate_k"]),
                    "batch_size": second.number_input("重排 · 批大小", min_value=1, value=reranker["batch_size"]),
                }
                st.caption(f"后端 · {reranker['backend']}　|　模型文件 · {reranker.get('model_file') or '默认'}")
        submitted = st.form_submit_button("应用检索设置", type="primary")
    if submitted:
        if "reranker" in overrides and (not overrides["reranker"]["model"]
                                       or runtime.contains_api_key(overrides["reranker"]["model"])):
            st.error("请输入有效的重排模型名称。")
            return
        state.strategy = strategy
        state.top_k = top_k
        state.overrides = {**state.overrides, strategy: overrides}
        state.settings_saved = True
        st.rerun()
    if state.pop("settings_saved", False):
        st.success("检索设置已应用，下一次请求将使用这组参数。")


def get_retriever(config, documents):
    index_config = {key: value for key, value in config.items()
                    if key in {"docs_path", "chunk_size", "chunk_overlap", "retriever",
                               "bm25", "embedding", "hybrid", "reranker"}}
    return cached_retriever(tuple(documents), index_config)


def evidence_title(row, titles):
    source = titles.get(row["source"], row["source"])
    if "chunk_id" not in row:
        return f"{source} · 原文补读 {row['start']}–{row['end']}"
    return f"{source} · {row['chunk_id']}"


def evidence_text(row):
    if "score" in row:
        st.caption(f"排名 {row['rank']} · 原始相关性分数 {row['score']!r}")
    st.text(row["text"])
    if row.get("text_truncated"):
        st.caption("此工具返回的片段达到长度上限。")


def apply_debug_strategy():
    st.session_state.strategy = st.session_state._debug_strategy


def render_debug(config, documents, titles, issue):
    st.title(DEBUG)
    state = st.session_state
    strategy = runtime.STRATEGIES[state.strategy].label
    strategy_col, settings_col = st.columns([3, 2], vertical_alignment="bottom")
    strategy_col.selectbox("检索策略", list(runtime.STRATEGIES), index=list(runtime.STRATEGIES).index(state.strategy),
                           format_func=lambda key: runtime.STRATEGIES[key].label, key="_debug_strategy",
                           on_change=apply_debug_strategy, help="切换后，问答测试使用同一策略。")
    settings_col.button("修改检索设置", on_click=navigate, args=(SETTINGS,))
    st.caption(f"Top-K · {config['top_k']}")
    if issue:
        st.warning(issue)
    with st.form("debug"):
        query = st.text_input("检索问题", placeholder="例如：学生优惠票可以购买哪些席别？", max_chars=2000)
        submit = st.form_submit_button("开始检索", type="primary", disabled=bool(issue))
    if submit and query.strip():
        state.pop("debug_result", None)
        with st.spinner("正在准备索引并检索…"):
            try:
                started = perf_counter()
                retriever = get_retriever(config, documents)
                prepared = perf_counter()
                hits = retriever.search(query.strip(), config["top_k"])
                state.debug_result = (strategy, config["top_k"], runtime.evidence_rows(hits),
                                      perf_counter() - prepared, prepared - started)
            except Exception:
                st.error("检索失败。请检查模型依赖、权重缓存和高级参数，或切换为 BM25。")
    elif submit:
        st.warning("请输入检索问题。")
    if "debug_result" in state:
        run_strategy, top_k, rows, elapsed, *preparation = state.debug_result
        st.subheader("检索结果")
        timing = f"准备索引 {preparation[0]:.3f} 秒 · 检索 {elapsed:.3f} 秒" if preparation else f"{elapsed:.2f} 秒"
        st.caption(f"{run_strategy} · Top-K {top_k} · 命中 {len(rows)} 条 · {timing}。"
                   "分数保留检索器原值，各策略的分数尺度不同。")
        for row in rows:
            with st.expander(f"第 {row['rank']} 名 · {evidence_title(row, titles)}"):
                evidence_text(row)
        if not rows:
            st.info("未检索到匹配片段。")


def render_trace(events):
    if not events:
        st.caption("当前请求没有可显示的轨迹事件。")
    for row in events:
        event = row["event"]
        if event == "request_finished":
            continue
        parts = [EVENT_NAMES[event]]
        if "model_call" in row:
            parts.append(f"第 {row['model_call']} 次")
        if "elapsed_ms" in row:
            parts.append(f"{row['elapsed_ms']:.1f} 毫秒")
        if "error_type" in row:
            parts.append("请求超时" if row["error_type"] == "TimeoutError" else row["error_type"])
        if "tool" in row:
            parts.append(row["tool"])
        if "top_k" in row:
            parts.append(f"Top-K {row['top_k']}")
        if "count" in row:
            parts.append(f"命中 {row['count']} 条")
        if "ok" in row:
            parts.append("成功" if row["ok"] else "失败")
        if "allow_tools" in row:
            parts.append("允许调用工具" if row["allow_tools"] else "工具已禁用")
        if row.get("finish_reason"):
            parts.append(f"模型结束标记 {row['finish_reason']}")
        arguments = row.get("arguments")
        if arguments is not None and row["tool"] == "search_knowledge":
            top_k = arguments.get("top_k", 3)
            if type(top_k) is int and 1 <= top_k <= 10:
                suffix = "（工具默认）" if "top_k" not in arguments else ""
                parts.append(f"请求 Top-K {top_k}{suffix}")
        st.text(" · ".join(parts))
        if arguments is not None:
            st.json(arguments, expanded=False)
        if "observation" in row:
            observation = row["observation"]
            if observation.get("ok"):
                payload = observation["result"]
                if "count" in payload:
                    st.caption(f"实际返回 {payload['count']} 条检索证据。")
                st.json(payload, expanded=False)
            else:
                error = observation.get("error") or {}
                st.warning(TOOL_ERRORS.get(error.get("code"), "工具失败"))
                st.json(error, expanded=False)
        if row.get("usage"):
            st.caption(f"本次调用 Token · {row['usage']['total_tokens']}")


def render_result(result, mode, strategy, events, titles):
    reason = result.get("stop_reason", "")
    with st.chat_message("assistant"):
        if result["status"] in {"failed", "stopped"}:
            st.warning(STOP_REASONS.get(reason, reason or "请求未完成"))
        if result["answer"]:
            st.markdown(result["answer"])
        st.caption(f"{runtime.MODE_LABELS[mode]} · {strategy} · {STATUS[result['status']]} · "
                   f"{result['elapsed_seconds']:.2f} 秒")
    with st.expander("引用证据"):
        if result["citations"]:
            for row in result["citations"]:
                st.text(f"[{row['citation_id']}] {evidence_title(row, titles)}")
                excerpt = row["text"]
                st.caption(excerpt[:160] + ("…" if len(excerpt) > 160 else ""))
        else:
            st.caption("本次回答没有引用证据。")
    with st.expander("查看检索证据"):
        if not result["evidence"]:
            st.caption("本次请求没有取得证据。")
        for index, row in enumerate(result["evidence"], 1):
            st.text(f"[{row.get('citation_id', index)}] {evidence_title(row, titles)}")
            evidence_text(row)
    with st.expander("查看 Agent 执行过程" if mode != "rag" else "查看问答执行过程"):
        render_trace(events)
        st.caption(f"停止原因 · {STOP_REASONS.get(reason, reason or '未记录')}")
    with st.expander("查看运行统计"):
        st.text(f"请求 ID · {result['request_id']}")
        if "index_seconds" in result:
            st.text(f"准备索引 · {result['index_seconds']:.3f} 秒")
        retrieval_times = [row["elapsed_ms"] for row in events if "elapsed_ms" in row and (
            row["event"] == "retrieval_finished" or
            (row["event"] == "tool_finished" and row.get("tool") == "search_knowledge")
        )]
        model_times = [row["elapsed_ms"] for row in events if "elapsed_ms" in row
                       and row["event"] in {"model_finished", "model_failed"}]
        if retrieval_times:
            st.text(f"检索耗时 · {sum(retrieval_times):.1f} 毫秒")
        if model_times:
            st.text(f"模型调用耗时 · {sum(model_times) / 1000:.3f} 秒")
        st.text(f"模型调用 · {result['model_calls']} 次")
        finished = next((row for row in events if row["event"] == "request_finished"), {})
        if "tool_calls" in finished:
            st.text(f"工具执行 · {finished['tool_calls']} 次 · 失败 {finished['failed_calls']} 次")
        if "tool_attempts" in result:
            st.text(f"工具请求尝试 · {result['tool_attempts']} 次")
        if usage := result.get("token_usage"):
            st.text(f"Token · 输入 {usage['prompt_tokens']} · 输出 {usage['completion_tokens']} · 合计 {usage['total_tokens']}")
        elif known := result.get("known_token_usage", {}).get("total_tokens"):
            st.text(f"Token · 已知 {known}，部分模型调用缺少用量")
        else:
            st.text("Token · 未取得完整用量")
        st.caption(f"本地轨迹 · results/studio_traces/{Path(result['trace_path']).name}")


def render_qa(config, documents, titles, issue):
    st.title(QA)
    state = st.session_state
    state.setdefault("_mode", state.mode)
    mode = st.radio("执行模式", list(runtime.MODE_LABELS), format_func=runtime.MODE_LABELS.__getitem__,
                    horizontal=True, key="_mode")
    state.mode = mode
    strategy = runtime.STRATEGIES[state.strategy].label
    st.caption(f"12306 铁路规则　|　{strategy}　|　界面 Top-K · {config['top_k']}　|　回答风格 · 通俗简洁")
    if mode != "rag":
        st.caption("Agent 可指定 Top-K，省略时使用工具默认值 3。每次检索的实际值见执行过程。")
    st.button("修改检索设置", on_click=navigate, args=(SETTINGS,))
    generation_issue = runtime.generation_issue(mode)
    for message in filter(None, (issue, generation_issue)):
        st.warning(message)
    state.setdefault("_question", state.question)
    question = st.text_area("问题", key="_question", on_change=remember_question,
                            placeholder="例如：新生如何通过线上购买学生票？", max_chars=2000).strip()
    state.setdefault("submission", 0)
    send = st.empty()
    if send.button("发送", type="primary", key=f"send_{state.submission}",
                   disabled=bool(issue or generation_issue or state.get("running"))):
        if not question:
            st.warning("请输入问题。")
        elif runtime.contains_api_key(question):
            st.error("问题中包含当前 API 密钥，请移除后再发送。")
        else:
            state.running = True
            state.submission += 1
            send.empty()
            state.pop("answer_result", None)
            state.pop("answer_error", None)
            try:
                with st.spinner("正在准备索引…"):
                    started = perf_counter()
                    retriever = get_retriever(config, documents)
                    prepared = perf_counter()
                with st.spinner("正在检索并生成回答…"):
                    result = runtime.run_answer(mode, question, config, documents, retriever, runtime.TRACE_DIR)
                    result["elapsed_seconds"] = perf_counter() - started
                    result["index_seconds"] = prepared - started
                    try:
                        events = runtime.read_trace(Path(result["trace_path"]))
                    except (OSError, ValueError, KeyError):
                        events = []
                    state.answer_result = (result, mode, strategy, events, titles)
            except ImportError:
                state.answer_error = "依赖加载失败，请检查当前检索策略和问答模式的安装说明。"
            except Exception:
                state.answer_error = "运行失败。请检查模型权重、网络或 results 目录的写入权限。"
            finally:
                state.running = False
            st.rerun()
    if "answer_error" in state:
        st.error(state.answer_error)
    if "answer_result" in state:
        render_result(*state.answer_result)


def main():
    st.set_page_config(page_title="RailPolicy_RAG", layout="wide")
    st.markdown("""<style>
        .block-container { max-width: 76rem; padding-top: 3rem; }
        .stText pre { white-space: pre-wrap; overflow-wrap: anywhere; }
        </style>""", unsafe_allow_html=True)
    state = st.session_state
    for key, value in {"page": QA, "strategy": runtime.DEFAULT_STRATEGY, "top_k": None,
                       "overrides": {}, "question": "", "mode": "rag"}.items():
        state.setdefault(key, value)
    strategy = state.strategy
    try:
        config, documents = runtime.snapshot(strategy, state.overrides.get(strategy, {}), state.top_k)
        issue = runtime.retrieval_issue(strategy, config)
    except (OSError, KeyError, ValueError, TypeError):
        documents = []
        issue = "无法加载铁路知识库，请检查当前配置与资料目录。"
    with st.sidebar:
        st.title("RailPolicy_RAG")
        page = st.radio("功能导航", (QA, LIBRARY, DEBUG, SETTINGS), key="page", label_visibility="collapsed")
        st.divider()
        st.caption("当前知识库 · 12306 铁路规则")
        st.caption(f"当前检索器 · {runtime.STRATEGIES[strategy].label}")
        ready = not (issue or runtime.generation_issue(state.mode))
        st.caption(f"运行状态 · {'就绪' if ready else '未就绪'}")
    if page == SETTINGS:
        render_settings()
    elif not documents:
        st.error(issue)
    else:
        entries = load_catalog(project_path(runtime.config_path(strategy), config["docs_path"]), documents)
        titles = {entry.document.source: entry.title for entry in entries}
        if page == LIBRARY:
            render_catalog(entries)
        elif page == DEBUG:
            render_debug(config, documents, titles, issue)
        else:
            render_qa(config, documents, titles, issue)


if __name__ == "__main__":
    main()
