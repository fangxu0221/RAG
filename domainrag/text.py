"""轻量级文本规范化和中英文分词。

BM25 需要离散 token 才能统计词频；Embedding 模型直接接收原始文本，不走这里的
tokenize()。这里故意不用第三方中文分词库，以便清楚观察基线系统做了什么。
"""

from __future__ import annotations

import re
import unicodedata


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def normalize_text(text: str) -> str:
    """统一全角/半角等 Unicode 表示，并把连续空白压成一个空格。"""

    # NFKC 例如会把全角字母、数字转换为常见的兼容形式。
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """不依赖第三方库，对中英文混合文本进行透明、可解释的分词。

    英文统一转小写并按连续单词保留；中文连续片段拆为字级 bigram（二元组），
    短中文片段还会整体保留。这只是 BM25 基线，不是生产级分词器。
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalized):
        term = match.group(0)
        if term[0].isascii():
            # 英文/数字串直接作为一个 token，例如 "BM25" -> "bm25"。
            tokens.append(term)
            continue

        if len(term) == 1:
            tokens.append(term)
            continue

        # “关键词检索”会产生“关键、键词、词检、检索”，提升中文局部匹配能力。
        tokens.extend(term[index : index + 2] for index in range(len(term) - 1))
        if len(term) <= 6:
            tokens.append(term)
    return tokens
