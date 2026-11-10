"""冻结评测入口：先验清单、保留运行指纹、逐题落盘，禁止覆盖旧报告。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .infrastructure.llm.answer_judge import DeepSeekAnswerJudge
from .application.evaluation_service import evaluate_retrieval, evaluate_answers, summarize_answers
from .bootstrap import build_retriever, project_path
from .frozen import runtime_fingerprint, verify_frozen_dataset
from .infrastructure.llm.generator import DeepSeekGenerator
from .io import load_evaluation_cases, write_json


def run(config_path: Path, output: Path, answers: bool = False) -> dict[str, object]:
    dataset = verify_frozen_dataset(config_path)
    if dataset is None:
        raise ValueError("holdout requires a frozen dataset config")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "dataset": dataset, "runtime": runtime_fingerprint(),
        "answer_evaluation_status": "pending" if answers else "not_run",
    }
    # 独占创建，避免误覆盖已有报告；模型初始化失败也留下运行记录。
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    started = perf_counter()
    completed = []
    stage = "build_retriever"
    try:
        retriever, config = build_retriever(config_path)
        report["config"] = config
        cases = load_evaluation_cases(project_path(config_path, config["eval_path"]))
        report["requested_case_ids"] = [case.case_id for case in cases]
        report["retrieval"] = evaluate_retrieval(retriever, cases, top_k=int(config["top_k"]))
        write_json(output, report)
        if answers:
            stage = "answer_evaluation"
            # 本轮预注册模型；不根据测试集成绩切换模型或提示。
            generator = DeepSeekGenerator()
            judge = DeepSeekAnswerJudge()
            report["models"] = {"generator": generator.model, "judge": judge.model,
                                "temperature": 0, "generator_max_tokens": generator.max_tokens,
                                "judge_max_tokens": judge.max_tokens}

            def checkpoint(row: dict[str, object]) -> None:
                completed.append(row)
                report["answers"] = summarize_answers(completed)
                write_json(output, report)
                print(f"Completed {len(completed)}/{len(cases)}: {row['id']}", flush=True)

            report["answers"] = evaluate_answers(retriever, cases, generator, judge,
                                                  top_k=int(config["top_k"]), on_case=checkpoint)
            report["answer_evaluation_status"] = "complete"
        # 运行中发生资料改动同样不能标为成功。
        verify_frozen_dataset(config_path)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        if answers:
            report["answer_evaluation_status"] = "incomplete"
            report["unevaluated_case_ids"] = [cid for cid in dataset["case_ids"] if cid not in {row["id"] for row in completed}]
        raise
    finally:
        report["wall_seconds"] = round(perf_counter() - started, 4)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output, report)
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--answers", action="store_true", help="also run 30 generation/Judge pairs using configured API key")
    args = parser.parse_args()
    report = run(args.config, args.output, args.answers)
    print(json.dumps(report["retrieval"]["summary"], ensure_ascii=False, indent=2))
    if "answers" in report:
        print(json.dumps(report["answers"]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
