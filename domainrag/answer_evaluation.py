"""旧答案评测导入路径。"""

from .application.evaluation_service import (
    CITATION_PATTERN, SCORE_FIELDS, evaluate_answers, evaluate_citations, summarize_answers,
)
from .domain.protocols import AnswerGenerator, AnswerJudge, CitationAssessment, SemanticJudgement
from .infrastructure.llm.answer_judge import (
    JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE, DeepSeekAnswerJudge, build_judge_prompt,
)

__all__ = [
    "CITATION_PATTERN", "SCORE_FIELDS", "evaluate_answers", "evaluate_citations",
    "summarize_answers", "AnswerGenerator", "AnswerJudge", "CitationAssessment",
    "SemanticJudgement", "JUDGE_SYSTEM_PROMPT", "JUDGE_USER_TEMPLATE",
    "DeepSeekAnswerJudge", "build_judge_prompt",
]
