"""DeepSeek 原生 Tool Calling HTTP 适配器。"""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from ...domain.protocols import TOOL_SCHEMAS


class DeepSeekToolModel:
    """非流式Chat Completions；密钥只放请求头，日志不包含HTTP头。"""

    def __init__(self, model: str = "deepseek-v4-flash", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")

    def complete(self, messages: list[dict], *, allow_tools: bool, timeout: float) -> dict:
        payload = {"model": self.model, "messages": messages, "tools": TOOL_SCHEMAS,
                   "tool_choice": "auto" if allow_tools else "none",
                   "thinking": {"type": "disabled"}, "temperature": 0,
                   "max_tokens": 1024, "stream": False}
        request = Request("https://api.deepseek.com/chat/completions",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
