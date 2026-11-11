"""从当前检索快照及可选索引构建只读知识库目录。"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from domainrag.models import Document
from domainrag.text import normalize_text


CATEGORIES = {"faq": "常见问题", "notices": "使用须知", "regulations": "相关规程",
              "other": "其他资料"}
TITLE_ALIASES = {"faq.student_tickets": "学生票", "faq.children_tickets": "儿童票"}


@dataclass(frozen=True)
class Unit:
    id: str
    title: str
    text: str
    chapter: str
    section: str


@dataclass(frozen=True)
class CatalogEntry:
    document: Document
    title: str
    category: str
    url: str
    units: tuple[Unit, ...]


def _metadata(path: Path) -> list[dict]:
    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            records = json.loads(content)["sources"]
        return [row for row in records if isinstance(row, dict)] if isinstance(records, list) else []
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _unit_from_snapshot(document: Document, row: dict) -> Unit | None:
    start, end = row.get("markdown_start"), row.get("markdown_end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document.text):
        return None
    heading, _, body = document.text[start:end].partition("\n")
    title = re.sub(r"^#{1,6}\s+", "", heading).strip()
    unit_id = row.get("unit_id")
    recorded_text = row.get("text")
    if not isinstance(unit_id, str) or not isinstance(recorded_text, str):
        return None
    if normalize_text(f"{title}\n{body}") != normalize_text(recorded_text):
        return None
    return Unit(unit_id, title, body.strip(), str(row.get("chapter") or ""),
                str(row.get("section") or ""))


def load_catalog(docs_root: Path, documents: list[Document]) -> list[CatalogEntry]:
    sources = {str(row.get("markdown_path")): row for row in _metadata(docs_root / "sources.json")}
    unit_rows = defaultdict(list)
    for row in _metadata(docs_root / "units.jsonl"):
        unit_rows[str(row.get("source"))].append(row)

    entries = []
    for doc in documents:
        meta = sources.get(doc.source, {})
        heading = re.search(r"^#\s+(.+)$", doc.text, re.MULTILINE)
        title = str(TITLE_ALIASES.get(str(meta.get("source_id"))) or meta.get("title")
                    or (heading[1].strip() if heading else Path(doc.source).stem))
        category = str(meta.get("category") or doc.source.split("/")[0])
        url = str(meta.get("url") or "")
        try:
            parsed = urlsplit(url)
            official_url = url if parsed.scheme == "https" and (
                parsed.hostname == "12306.cn" or (parsed.hostname or "").endswith(".12306.cn")
            ) else ""
        except ValueError:
            official_url = ""
        units = tuple(dict.fromkeys(unit for row in unit_rows[doc.source]
                                    if (unit := _unit_from_snapshot(doc, row)) is not None))
        entries.append(CatalogEntry(doc, title, category if category in CATEGORIES else "other",
                                    official_url, units))
    return entries


def filter_units(entry: CatalogEntry, keyword: str) -> list[Unit]:
    keyword = normalize_text(keyword).casefold()
    if entry.category == "regulations":
        return [unit for unit in entry.units if keyword in normalize_text(
            " ".join((unit.title, unit.chapter, unit.section, unit.text))).casefold()]
    return [unit for unit in entry.units if keyword in normalize_text(unit.title).casefold()]


def question_for(entry: CatalogEntry, unit: Unit) -> str:
    if entry.category == "faq":
        return re.sub(r"^\d+\.\s*", "", unit.title)
    return f"《{entry.title}》{unit.title}是怎么规定的？"
