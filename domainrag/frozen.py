"""冻结测试集的完整性检查与运行指纹；检查失败时禁止继续评测。"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from .io import load_documents, load_evaluation_cases
from .generator import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def text_sha256(path: Path) -> str:
    # 跨 Windows/Git 换行转换保持相同内容指纹；不忽略其他空白变化。
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("frozen path escapes project root")
    return path


def verify_frozen_dataset(config_path: Path) -> dict[str, object] | None:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "freeze_manifest" not in config:
        return None
    manifest_path = _inside(root, config["freeze_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config_path.relative_to(root).as_posix() not in manifest["config_paths"]:
        raise ValueError("config is not registered in frozen manifest")
    for field in ("docs_path", "eval_path"):
        if config[field] != manifest[field]:
            raise ValueError(f"frozen {field} differs from manifest")
    expected = manifest["files"]
    required = set(manifest["config_paths"]) | {manifest["eval_path"], manifest["protocol_path"]}
    required.update(p.relative_to(root).as_posix() for p in _inside(root, config["docs_path"]).rglob("*") if p.is_file())
    if required != set(expected):
        raise ValueError("frozen inventory mismatch (added/removed/unlisted files)")
    for name, digest in expected.items():
        path = _inside(root, name)
        if not path.is_file() or text_sha256(path) != digest:
            raise ValueError(f"frozen content changed: {name}; create a new dataset version")
    prompt_digest = hashlib.sha256((SYSTEM_PROMPT + USER_PROMPT_TEMPLATE).encode("utf-8")).hexdigest()
    if prompt_digest != manifest["generator_prompt_sha256"]:
        raise ValueError("generation prompt differs from frozen baseline")
    documents = {doc.source: doc.text for doc in load_documents(_inside(root, config["docs_path"]))}
    cases = load_evaluation_cases(_inside(root, config["eval_path"]))
    ids, questions = set(), set()
    for case in cases:
        question = "".join(case.question.split()).casefold()
        if not question or case.case_id in ids or question in questions:
            raise ValueError("duplicate/empty frozen question or id")
        ids.add(case.case_id)
        questions.add(question)
        if not case.expected_answer or case.metadata.get("split") != "test":
            raise ValueError(f"missing reference or wrong split: {case.case_id}")
        evidence = case.metadata.get("evidence", [])
        if case.answerable:
            if set(case.relevant_sources) != {item["source"] for item in evidence}:
                raise ValueError(f"incomplete evidence sources: {case.case_id}")
            for item in evidence:
                if not item["quote"] or item["quote"] not in documents.get(item["source"], ""):
                    raise ValueError(f"evidence quote not found: {case.case_id}")
            if case.metadata["question_type"] == "multi_source" and len(set(case.relevant_sources)) < 2:
                raise ValueError(f"multi_source requires at least two documents: {case.case_id}")
        elif evidence or not case.metadata.get("unanswerable_reason"):
            raise ValueError(f"unanswerable case requires absence rationale and no supporting evidence: {case.case_id}")
    old_questions = {
        "".join(case.question.split()).casefold()
        for case in load_evaluation_cases(_inside(root, manifest["development_eval_path"]))
    }
    if questions.intersection(old_questions):
        raise ValueError("holdout question duplicates development set")
    return {
        "dataset_id": manifest["dataset_id"],
        "frozen_at": manifest["frozen_at"],
        "manifest_sha256": text_sha256(manifest_path),
        "files": expected,
        "generator_prompt_sha256": prompt_digest,
        "case_ids": [case.case_id for case in cases],
        "review_status": manifest["review_status"],
    }


def runtime_fingerprint() -> dict[str, object]:
    package = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {path.name: text_sha256(path) for path in sorted(package.glob("*.py"))},
    }
