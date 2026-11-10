"""DeepSeek 答案评审与响应解析。"""

from __future__ import annotations

import json
import os
from time import perf_counter
from urllib.request import Request, urlopen

from ...domain.protocols import CitationAssessment, SemanticJudgement
from ...models import EvaluationCase, SearchResult
from .generator import api_call_stats, build_user_prompt


JUDGE_SYSTEM_PROMPT = """你是 RAG 系统的严格评测器。请只输出 JSON 对象，不要输出 Markdown。

分别评估：
1. correctness：回答与参考答案在核心事实上是否一致。
2. faithfulness：回答中的事实性主张是否都能由检索资料支持。
3. citation_assessments：逐个判断回答中的有效引用编号所对应的 Chunk，是否直接支持该引用关联的核心结论。即使两个 Chunk 来自同一文档，无关、只支持次要旁支或多余的 Chunk 也必须标为 false。如果回答声称“资料无法确定”，但被引用 Chunk 实际已经给出答案，该引用必须标为 false。
4. abstained：布尔值。仅当回答明确表示无法确定问题所需结论且没有猜测该结论时为 true；“无法确定，但是答案可能是X”不是有效拒答。不要仅靠关键词判断。

标注不可回答的题目以明确拒答且不编造为正确行为。正确拒答不要求引用；不得因缺少引用而扣 correctness 或 faithfulness。仍须检查拒答中附带的事实是否有依据。标注可回答但当前检索漏证据时，拒答可以忠实，但不算回答正确。

两项都只能给整数 0、1、2：
- 0：错误、缺失或存在明显无依据内容。
- 1：部分正确/大部分有依据，但有重要遗漏或轻微无依据扩展。
- 2：核心结论完整正确/所有事实主张均有资料支持。

常识上看似合理但资料没有陈述、也不是核心结论所必需的一步推断，仍属于无依据扩展，应降低 faithfulness；对应引用也不能用来支持该扩展。

只判断回答中出现且存在于检索资料里的引用编号；不存在的编号由程序单独处理。
输出格式必须为：
{"correctness": 2, "faithfulness": 2, "abstained": false, "reason": "简短理由", "citation_assessments": [{"index": 1, "supported": true, "reason": "该 Chunk 直接支持对应结论"}]}
""".strip()

JUDGE_USER_TEMPLATE = """问题：
{question}

参考答案：
{expected_answer}

根据完整知识库标注的可回答性：{answerable}

检索资料：
{context}

待评估回答：
{answer}
"""


def build_judge_prompt(
    case: EvaluationCase,
    answer: str,
    results: list[SearchResult],
) -> str:
    """组合问题、参考答案、检索上下文和待评回答，供 Judge 比较。"""

    if not case.expected_answer:
        raise ValueError(f"case {case.case_id!r} has no expected_answer")
    # 复用生成器的上下文编号格式，但去掉末尾重复的“问题”部分。
    context = build_user_prompt(case.question, results).split("\n\n问题：", 1)[0]
    return JUDGE_USER_TEMPLATE.format(
        question=case.question,
        expected_answer=case.expected_answer,
        context=context,
        answer=answer,
        answerable=case.answerable,
    )


def _validated_score(value: object, name: str) -> int:
    """严格验证 Judge 分数只能是整数 0/1/2。"""

    # Python 中 bool 是 int 的子类，因此必须先排除 True/False。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"judge field {name!r} must be an integer from 0 to 2")
    score = int(value)
    if score != value or score not in {0, 1, 2}:
        raise ValueError(f"judge field {name!r} must be an integer from 0 to 2")
    return score


def _parse_citation_assessments(
    value: object,
) -> tuple[CitationAssessment, ...]:
    """把 Judge 的 JSON 数组转换成强类型 CitationAssessment 元组。"""

    if not isinstance(value, list):
        raise ValueError("judge field 'citation_assessments' must be a list")
    assessments: list[CitationAssessment] = []
    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"citation assessment {position} must be a JSON object"
            )
        # index 必须是从 1 开始的正整数，和 Prompt 中的 [1]、[2] 对齐。
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, (int, float)):
            raise ValueError(f"citation assessment {position} has invalid index")
        parsed_index = int(index)
        if parsed_index != index or parsed_index <= 0:
            raise ValueError(f"citation assessment {position} has invalid index")
        supported = item.get("supported")
        if not isinstance(supported, bool):
            raise ValueError(
                f"citation assessment {position} has invalid supported flag"
            )
        assessments.append(
            CitationAssessment(
                index=parsed_index,
                supported=supported,
                reason=str(item.get("reason", "")),
            )
        )
    return tuple(assessments)


class DeepSeekAnswerJudge:
    """使用 DeepSeek JSON Output 评价正确性、忠实度和逐 Chunk 引用支持。"""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 512,
    ):
        # 与生成器相同：测试可注入 key，真实使用则从环境变量读取。
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for answer judging")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = resolved_api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def judge(
        self,
        case: EvaluationCase,
        answer: str,
        results: list[SearchResult],
    ) -> SemanticJudgement:
        """调用 Judge 模型，并将其嵌套 JSON 输出解析为 SemanticJudgement。"""

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_judge_prompt(case, answer, results),
                    },
                ],
                "thinking": {"type": "disabled"},
                "temperature": 0,
                # 要求模型输出 JSON，便于程序稳定解析而不是处理自然语言格式。
                "response_format": {"type": "json_object"},
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
            response_payload = json.loads(response.read().decode("utf-8"))
        latency_seconds = perf_counter() - started_at
        # 第一层 JSON 是 API 响应；content 字符串内部是 Judge 生成的第二层 JSON。
        content = response_payload["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("judge returned empty JSON content")
        judgement = json.loads(content)
        if not isinstance(judgement.get("abstained"), bool):
            raise ValueError("judge field 'abstained' must be a boolean")
        return SemanticJudgement(
            correctness=_validated_score(judgement.get("correctness"), "correctness"),
            faithfulness=_validated_score(
                judgement.get("faithfulness"), "faithfulness"
            ),
            reason=str(judgement.get("reason", "")),
            citation_assessments=_parse_citation_assessments(
                judgement.get("citation_assessments")
            ),
            stats=api_call_stats(response_payload, latency_seconds),
            abstained=judgement["abstained"],
        )
