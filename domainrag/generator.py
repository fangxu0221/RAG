"""旧生成器导入路径。"""

from .domain.protocols import ApiCallStats, GenerationResult
from .infrastructure.llm.generator import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, DeepSeekGenerator, OllamaGenerator,
    api_call_stats, build_user_prompt,
)

__all__ = [
    "ApiCallStats", "GenerationResult", "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE",
    "DeepSeekGenerator", "OllamaGenerator", "api_call_stats", "build_user_prompt",
]
