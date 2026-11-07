"""Collect nine explicitly selected public 12306 documents, preserving raw evidence.

Standalone optional utility: requires lxml; never executes downloaded JavaScript.
It neither changes RAG configuration nor invokes a model nor publishes a commit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from lxml import etree, html


PROJECT = Path(__file__).resolve().parents[1]
HELP_URL = "https://kyfw.12306.cn/otn/gonggao/help.html"
HOSTS = {"kyfw.12306.cn", "mobile.12306.cn"}
FAQS = [
    ("onlineBooking", "互联网购票", "online_booking", "购票"),
    ("windowEndorse", "改签", "ticket_changes", "车票变更"),
    ("windowRefund", "售票窗口退票", "window_refunds", "车票变更"),
    ("onlineRefund", "互联网退票", "online_refunds", "车票变更"),
    ("student", "学生", "student_tickets", "特殊旅客服务"),
    ("children", "儿童", "children_tickets", "特殊旅客服务"),
]
NOTICES = [
    ("buyTicketsInstruction", "铁路旅客购票须知", "ticket_purchase_notice"),
    ("travelInstruction", "铁路旅客旅行须知", "travel_notice"),
]
REGULATION_TITLE = "中国国家铁路集团有限公司铁路旅客运输规程"
REGULATION_URL = "https://kyfw.12306.cn/otn/gonggao/saleTicketMeans.html?linktypeid=means2"
CLASS_Q = "contains(concat(' ',normalize-space(@class),' '),' q-list ')"
CN = "一二三四五六七八九十百零〇两"
ARTICLE = re.compile(rf"^(第([{CN}]+)条)\s*")
SECTION = re.compile(rf"^第[{CN}]+节")
CHAPTER = re.compile(rf"^第[{CN}]+章")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def canonical(value: str) -> str:
    return re.sub(r"\s+", "", value)


def cn_number(value: str) -> int:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
              "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    total = current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        else:
            total += (current or 1) * {"十": 10, "百": 100}[char]
            current = 0
    return total + current


def text_lines(element) -> list[str]:
    """HTML block/br boundaries only; comments, scripts and styles are excluded."""
    parts: list[str] = []

    def visit(node):
        if not isinstance(node.tag, str) or node.tag.lower() in {"script", "style"}:
            return
        tag = node.tag.lower()
        block = tag in {"p", "div", "li", "dt", "dd", "h1", "h2", "h3", "h4", "ul", "ol"}
        if block or tag == "br":
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                parts.append(child.tail)
        if block:
            parts.append("\n")

    visit(element)
    # Whitespace-only lines are layout, not additional evidence paragraphs.
    return [line for raw in "".join(parts).splitlines() if (line := normalize(raw))]


def parse_html(raw: bytes):
    return html.fromstring(raw.decode("utf-8", errors="strict"))


def faq_units(raw: bytes, source_id: str) -> list[dict]:
    tree = parse_html(raw)
    lists = tree.xpath(f"//*[{CLASS_Q}]")
    if len(lists) != 1:
        raise ValueError(f"{source_id}: expected exactly one FAQ list")
    if lists[0].xpath(".//table|.//img|.//iframe"):
        raise ValueError("FAQ contains non-text evidence; explicit extraction needed")
    units = []
    for item in lists[0].xpath("./li"):
        questions, answers = item.xpath("./dt"), item.xpath("./dd")
        # One official page contains an extra empty <li> before its fourth Q&A.
        if not questions and not answers and not normalize(item.text_content()):
            continue
        if len(questions) != 1 or len(answers) != 1:
            raise ValueError("FAQ question/answer pairing changed")
        number = len(units) + 1
        question = " ".join(text_lines(questions[0]))
        paragraphs = text_lines(answers[0])
        label = re.match(r"^(\d+)\s*[.．、]", question)
        if not question or not paragraphs or not label or int(label[1]) != number:
            raise ValueError("Empty or non-contiguous FAQ")
        expected = canonical("".join(answers[0].itertext()))
        if canonical("".join(paragraphs)) != expected:
            raise ValueError("FAQ answer text was lost in conversion")
        units.append({"unit_id": f"{source_id}.q{number:03d}", "kind": "faq",
                      "number": number, "heading": question, "paragraphs": paragraphs})
    if not units:
        raise ValueError("Empty FAQ page")
    return units


def notice_units(raw: bytes, source_id: str) -> list[dict]:
    """Decode the literal JSON array backing Vue's rules; no JS evaluation."""
    script = raw.decode("utf-8", errors="strict")
    candidates = []
    for match in re.finditer(r"var\s+([A-Za-z_$][\w$]*)\s*=\s*(\[\[)", script):
        start = match.start(2)
        try:
            values, end = json.JSONDecoder().raw_decode(script, start)
        except json.JSONDecodeError:
            continue
        suffix = script[end:]
        if not re.match(r";\s*[A-Za-z_$][\w$]*\.default=" + re.escape(match[1]), suffix):
            continue
        if isinstance(values, list) and all(isinstance(group, list) and group and
                all(isinstance(p, str) for p in group) for group in values):
            candidates.append(values)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one literal notice data array")
    units = []
    for number, paragraphs in enumerate(candidates[0], 1):
        label = re.match(r"^(\d+)\s*[.．、]", paragraphs[0])
        if not label or int(label[1]) != number or not all(p.strip() for p in paragraphs):
            raise ValueError("Notice paragraph numbering changed")
        units.append({"unit_id": f"{source_id}.item{number:03d}", "kind": "notice_item",
                      "number": number, "heading": f"第{number}项",
                      "paragraphs": paragraphs})
    return units


def regulation_units(raw: bytes, source_id: str) -> tuple[list[dict], list[dict]]:
    tree = parse_html(raw)
    elements = tree.xpath("//*[@id='means2-box']")
    if len(elements) != 1:
        raise ValueError("Regulation selector changed")
    element = elements[0]
    title = normalize("".join(element.xpath("./h1")[0].itertext()))
    if canonical(title) != REGULATION_TITLE:
        raise ValueError(f"Wrong regulation: {title}")
    if element.xpath(".//table|.//img|.//iframe"):
        raise ValueError("Regulation contains non-text evidence")
    units, events = [], []
    chapter = section = None
    current = None
    for child in element:
        if not isinstance(child.tag, str) or child.tag == "h1":
            continue
        paragraphs = text_lines(child)
        if not paragraphs:
            continue
        text = " ".join(paragraphs)
        if CHAPTER.match(text) or SECTION.match(text):
            current = None
            if CHAPTER.match(text):
                chapter, section = text, None
                events.append({"level": 2, "heading": text})
            else:
                section = text
                events.append({"level": 3, "heading": text})
            continue
        match = ARTICLE.match(paragraphs[0])
        if match:
            number = cn_number(match[2])
            body = [paragraphs[0][match.end():], *paragraphs[1:]]
            current = {"unit_id": f"{source_id}.article{number:03d}", "kind": "article",
                       "number": number, "heading": match[1], "chapter": chapter,
                       "section": section, "paragraphs": [p for p in body if p]}
            units.append(current)
            events.append({"unit": current})
        elif current is not None:
            current["paragraphs"].extend(paragraphs)
        else:
            raise ValueError(f"Unassigned regulation text: {text[:80]}")
    numbers = [u["number"] for u in units]
    if not numbers or numbers != list(range(1, max(numbers) + 1)):
        raise ValueError("Regulation articles missing/duplicated/out of order")
    # Compare all non-title body text, including chapter/section headings.
    expected = canonical("".join("".join(child.itertext()) for child in element
                                  if isinstance(child.tag, str) and child.tag != "h1"))
    converted = canonical("".join((e.get("heading") or
                     e["unit"]["heading"] + "".join(e["unit"]["paragraphs"])) for e in events))
    if expected != converted:
        raise ValueError("Regulation body text was lost in conversion")
    return units, events


class Collector:
    def __init__(self, output: Path):
        self.output = output.resolve()
        if (self.output / "sources.json").exists():
            raise FileExistsError("sources.json already exists; use a new --output for a new version")
        self.snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.raw_root = self.output / "raw" / self.snapshot_id
        self.raw_root.mkdir(parents=True)
        self.resources = []
        self.sources = []
        self.units = []

    def fetch(self, url: str, relative: str) -> tuple[bytes, dict]:
        if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in HOSTS:
            raise ValueError("Only allowlisted public 12306 HTTPS URLs are collected")
        request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/javascript,*/*"})
        with urlopen(request, timeout=30) as response:
            if urlsplit(response.url).hostname not in HOSTS or response.status != 200:
                raise ValueError("Unexpected redirect or HTTP status")
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Response exceeds bounded collection size")
            metadata = {"url": url, "final_url": response.url, "http_status": response.status,
                        "retrieved_at": utc_now(), "content_type": response.headers.get("Content-Type"),
                        "http_last_modified": response.headers.get("Last-Modified"),
                        "http_etag": response.headers.get("ETag")}
        path = self.raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        metadata.update(path=path.relative_to(self.output).as_posix(), bytes=len(raw), sha256=sha256(raw), encoding="utf-8")
        self.resources.append(metadata)
        time.sleep(0.2)
        return raw, metadata

    def add_document(self, source_id, title, category, relative, url, units, resources,
                     *, events=None, effective_date=None, effective_evidence=None, extra=None):
        destination = self.output / relative
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        retrieved_at = resources[0]["retrieved_at"]
        lines = [f"# {title}", "", f"来源：{url}", f"采集时间（UTC）：{retrieved_at}", ""]
        if events is None:
            events = [{"unit": unit} for unit in units]
        for event in events:
            if "unit" not in event:
                lines.extend(["#" * event["level"] + " " + event["heading"], ""])
                continue
            unit = event["unit"]
            level = 4 if category == "regulations" else 2
            rendered = "#" * level + " " + unit["heading"] + "\n\n" + "\n\n".join(unit["paragraphs"]) + "\n\n"
            prefix = "\n".join(lines)
            # Offsets are UTF-8-independent Python character offsets in this Markdown.
            start = len(prefix) + (1 if lines else 0)
            lines.extend(rendered.rstrip("\n").split("\n") + [""])
            evidence = {**unit, "source_id": source_id, "source": relative, "source_url": url,
                        "markdown_heading": unit["heading"], "markdown_start": start,
                        "markdown_end": start + len(rendered.rstrip("\n")),
                        "text": unit["heading"] + "\n" + "\n\n".join(unit["paragraphs"])}
            evidence["text_sha256"] = sha256(evidence["text"].encode("utf-8"))
            self.units.append(evidence)
        content = "\n".join(lines).rstrip() + "\n"
        destination.write_text(content, encoding="utf-8", newline="\n")
        body = "\n\n".join(u["heading"] + "\n" + "\n\n".join(u["paragraphs"]) for u in units)
        self.sources.append({
            "source_id": source_id, "title": title, "category": category,
            "url": url, "markdown_path": relative, "retrieved_at": retrieved_at,
            "publisher": "中国铁路12306 / 中国国家铁路集团有限公司",
            "published_at": None, "effective_date": effective_date,
            "effective_date_evidence": effective_evidence,
            "version_id": self.snapshot_id + "." + sha256(body.encode("utf-8"))[:12],
            "markdown_sha256": sha256(content.encode("utf-8")),
            "body_sha256": sha256(body.encode("utf-8")), "unit_count": len(units),
            "body_characters": len(body), "raw_resource_paths": [r["path"] for r in resources],
            "extraction_status": "complete", "consistency_review": "pending",
            "publication_date_note": "官网正文未明确标示发布日期；HTTP Last-Modified不作为发布日期或生效日期。",
            **(extra or {}),
        })
        print(f"Collected {title}: {len(units)} units", flush=True)

    def collect(self):
        help_raw, _ = self.fetch(HELP_URL, "discovery/help.html")
        help_tree = parse_html(help_raw)
        script_url = urljoin(HELP_URL, "../resources/js/help/help.js")
        script_raw, _ = self.fetch(script_url, "discovery/help.js")
        if 'this.id+".html"' not in script_raw.decode("utf-8"):
            raise ValueError("FAQ link-routing convention changed")
        for page_id, title, stem, group in FAQS:
            anchors = help_tree.xpath(f"//a[@id='{page_id}']")
            if len(anchors) != 1 or normalize(anchors[0].text_content()) != title:
                raise ValueError(f"FAQ discovery mismatch: {page_id}")
            url = urljoin(HELP_URL, page_id + ".html")
            raw, resource = self.fetch(url, f"pages/faq/{stem}.html")
            source_id = "faq." + stem
            self.add_document(source_id, title, "faq", f"faq/{stem}.md", url,
                              faq_units(raw, source_id), [resource], extra={"topic_group": group,
                              "extraction_method": "HTML .q-list > li > dt/dd"})
        for page_name, title, stem in NOTICES:
            anchors = help_tree.xpath("//a[contains(@href,'" + page_name + ".html')]")
            if len(anchors) != 1:
                raise ValueError("Notice link discovery mismatch")
            url = urljoin(HELP_URL, anchors[0].get("href"))
            raw, page_resource = self.fetch(url, f"pages/notices/{stem}.html")
            scripts = [src for src in parse_html(raw).xpath("//script[@src]/@src")
                       if re.search(r"(?:^|/)" + re.escape(page_name) + r"\.[^/]+\.js$", src)]
            if len(scripts) != 1:
                raise ValueError("Notice body asset discovery mismatch")
            asset_url = urljoin(url, scripts[0])
            asset, asset_resource = self.fetch(asset_url, f"assets/{stem}.js")
            source_id = "notice." + stem
            self.add_document(source_id, title, "notices", f"notices/{stem}.md", url,
                              notice_units(asset, source_id), [page_resource, asset_resource], extra={
                                  "extraction_method": "Official Vue bundle literal rules array; JSON decoded without JS execution",
                                  "body_asset_url": asset_url})
        raw, resource = self.fetch(REGULATION_URL, "pages/regulations/passenger_transport_rules.html")
        source_id = "regulation.cr_passenger_transport_rules"
        units, events = regulation_units(raw, source_id)
        last = units[-1]
        match = re.search(r"本规程自(\d{4})年(\d{1,2})月(\d{1,2})日起施行", "".join(last["paragraphs"]))
        if not match:
            raise ValueError("Regulation effective date needs manual review")
        effective_date = f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}"
        priority = [u for u in units if u["section"] and any(topic in u["section"] for topic in
                    ("售票与购票", "售票条件", "车票查验和安全检查", "乘车条件", "变更", "退票"))
                    and u["chapter"] and "旅客运输" in u["chapter"]]
        self.add_document(source_id, REGULATION_TITLE, "regulations", "regulations/passenger_transport_rules.md",
                          REGULATION_URL, units, [resource], events=events, effective_date=effective_date,
                          effective_evidence={"unit_id": last["unit_id"], "text": "".join(last["paragraphs"])}, extra={
                              "official_title": REGULATION_TITLE, "document_nature": "国铁集团企业客运规程；不是同名交通运输部规章全文",
                              "content_scope": "full_text", "extraction_method": "Only HTML #means2-box; other hidden rule documents excluded",
                              "priority_unit_ids": [u["unit_id"] for u in priority],
                              "priority_article_numbers": [u["number"] for u in priority]})
        with (self.output / "units.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for unit in self.units:
                stream.write(json.dumps(unit, ensure_ascii=False) + "\n")
        with (self.output / "eval.jsonl").open("x", encoding="utf-8"):
            pass
        manifest = {"schema_version": 1, "dataset_id": "railway_12306_v1", "snapshot_id": self.snapshot_id,
                    "created_at": utc_now(), "entry_url": HELP_URL, "status": "collected_unreviewed",
                    "scope": {"faq_pages": 6, "notices": 2, "regulations": 1},
                    "source_count": len(self.sources), "unit_count": len(self.units),
                    "original_configs_switched": False, "ingestion_root": "data/railway",
                    "supported_knowledge_files": "Only regulations/*.md, notices/*.md and faq/*.md are corpus text",
                    "units_path": "units.jsonl", "units_sha256": sha256((self.output / "units.jsonl").read_bytes()),
                    "evaluation": {"path": "eval.jsonl", "case_count": 0, "status": "empty_pending_annotation",
                                   "sha256": sha256(b"")},
                    "collector": {"path": "scripts/collect_railway_v1.py", "sha256": sha256(Path(__file__).read_bytes())},
                    "review_notes": ["采集时间不是发布时间或生效时间。", "保留各来源原文，跨来源规则一致性尚待审核。",
                                     "未构造评测题、未调用生成模型、未切换现有知识库配置。",
                                     "优先条款索引仅供后续范围选择；当前未实现检索排序权重。"],
                    "sources": self.sources, "raw_resources": self.resources}
        (self.output / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify(self.output)


def verify(output: Path):
    manifest = json.loads((output / "sources.json").read_text(encoding="utf-8"))
    for resource in manifest["raw_resources"]:
        if sha256((output / resource["path"]).read_bytes()) != resource["sha256"]:
            raise ValueError("Raw resource hash mismatch")
    units = [json.loads(line) for line in (output / "units.jsonl").read_text(encoding="utf-8").splitlines()]
    if sha256((output / "units.jsonl").read_bytes()) != manifest["units_sha256"]:
        raise ValueError("Evidence-unit hash mismatch")
    for source in manifest["sources"]:
        path = output / source["markdown_path"]
        content = path.read_text(encoding="utf-8")
        if sha256(path.read_bytes()) != source["markdown_sha256"]:
            raise ValueError("Markdown hash mismatch")
        matched = [u for u in units if u["source_id"] == source["source_id"]]
        if len(matched) != source["unit_count"]:
            raise ValueError("Evidence unit count mismatch")
        for unit in matched:
            span = content[unit["markdown_start"]:unit["markdown_end"]]
            expected = unit["heading"] + "\n\n" + "\n\n".join(unit["paragraphs"])
            if canonical(span.lstrip("# ")) != canonical(expected):
                raise ValueError(f"Markdown evidence offset mismatch: {unit['unit_id']}")
            if sha256(unit["text"].encode("utf-8")) != unit["text_sha256"]:
                raise ValueError("Evidence text hash mismatch")
    paths = {p.relative_to(output).as_posix() for p in output.rglob("*.md")}
    if paths != {s["markdown_path"] for s in manifest["sources"]} or len(paths) != 9:
        raise ValueError("Unexpected Markdown documents in corpus")
    print(f"Verified {len(paths)} Markdown documents, {len(units)} evidence units and {len(manifest['raw_resources'])} raw resources.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/railway")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
    else:
        Collector(args.output).collect()


if __name__ == "__main__":
    main()
