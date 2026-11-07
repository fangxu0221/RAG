"""把检索结果组织成 Prompt，并调用 Ollama 或 DeepSeek 生成带引用的答案。

本模块只负责“生成”，不判断答案好坏；正确性、忠实度和引用质量由
answer_evaluation.py 单独完成。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter
from urllib.request import Request, urlopen

from .models import SearchResult


SYSTEM_PROMPT = """你是一个严格依据资料回答问题的助手。
优先直接回答问题的核心结论。只要资料已经明确陈述答案，或可以通过一步直接推断得到答案，就应直接作答；不要因为资料没有说明额外后果、实现细节或使用与问题完全相同的措辞而回答“无法确定”。
回答应保持最小充分：给出问题所需的结论和资料明确支持的理由即可，不要为了显得完整而补充资料没有陈述的效果、后果或因果解释。
只有检索资料确实无法支持核心结论时，才回答“根据当前资料无法确定”，不要编造。
只列出真正支持核心结论的来源编号，不要引用只包含相似术语的资料。引用必须写成 `[1]`、`[2]` 这种格式。"""

USER_PROMPT_TEMPLATE = """资料：
{context}

问题：{question}
"""


@dataclass(frozen=True)
class ApiCallStats:
    """一次模型 API 调用的耗时与 Token 用量，供成本和性能分析使用。"""

    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    """生成文本及其 API 统计，避免只返回字符串后丢失运行证据。"""

    text: str
    stats: ApiCallStats = ApiCallStats()


def _usage_value(usage: dict[str, object], key: str) -> int:
    """安全读取 API usage 字段；字段缺失或类型异常时按 0 处理。"""

    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def api_call_stats(
    response_payload: dict[str, object],
    latency_seconds: float,
) -> ApiCallStats:
    """把 OpenAI 兼容接口的 usage JSON 转成统一的 ApiCallStats。"""

    usage_value = response_payload.get("usage", {})
    usage = usage_value if isinstance(usage_value, dict) else {}
    return ApiCallStats(
        latency_seconds=latency_seconds,
        prompt_tokens=_usage_value(usage, "prompt_tokens"),
        completion_tokens=_usage_value(usage, "completion_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
    )


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    """将 Top-K Chunk 编号后拼成上下文；答案中的 [1] 就指向第一个结果。"""

    # 编号顺序必须与 results 完全一致，引用评测才能用 index - 1 找回原 Chunk。
    context_parts = [
        f"[{index}] 来源={result.chunk.chunk_id}\n{result.chunk.text}"
        for index, result in enumerate(results, start=1)
    ]
    return USER_PROMPT_TEMPLATE.format(
        context="\n\n".join(context_parts),
        question=question,
    )


class OllamaGenerator:
    """通过本机 Ollama /api/generate 接口生成答案。"""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        """发送非流式请求，并把 Ollama 独有的 Token 字段转为统一统计。"""

        # Ollama generate 接口只有一个 prompt 字段，因此手动拼接系统与用户提示。
        prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(question, results)}"
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # 只统计实际 HTTP 调用时间，不把检索和 Prompt 拼接时间算进模型延迟。
        started_at = perf_counter()
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        latency_seconds = perf_counter() - started_at
        prompt_tokens = _usage_value(result, "prompt_eval_count")
        completion_tokens = _usage_value(result, "eval_count")
        return GenerationResult(
            text=str(result["response"]),
            stats=ApiCallStats(
                latency_seconds=latency_seconds,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class DeepSeekGenerator:
    """通过 DeepSeek Chat Completions API 生成严格依赖检索资料的答案。"""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 1024,
    ):
        # 优先允许测试显式注入 key；正常运行从环境变量读取，绝不写死在源码中。
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Define it in the current shell "
                "instead of writing the key into source code."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = resolved_api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GenerationResult:
        """发送 system/user 两角色消息并解析回答正文与 Token 统计。"""

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(question, results),
                    },
                ],
                # 评测需要尽量可复现，因此关闭思考模式并将 temperature 固定为 0。
                "thinking": {"type": "disabled"},
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started_at = perf_counter()
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        latency_seconds = perf_counter() - started_at
        # OpenAI 兼容格式的正文位于 choices[0].message.content。
        return GenerationResult(
            text=str(result["choices"][0]["message"]["content"]),
            stats=api_call_stats(result, latency_seconds),
        )
