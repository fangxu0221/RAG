"""固定配置的四组检索实验；复用CLI和评分器，原始结果仅写入新的本地目录。

先 --prepare 下载模型，再默认离线运行。没有生成、Judge或Agent入口。
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "results/railpolicy_v1/hf-home"
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

GROUPS = {
    "A": ("bm25_baseline.json", "railway_dev_bm25.json"),
    "B": ("embedding_baseline.json", "railway_dev_embedding.json"),
    "C": ("hybrid_rrf.json", "railway_dev_hybrid_rrf.json"),
    "D": ("hybrid_reranker.json", "railway_dev_hybrid_reranker.json"),
}
MODELS = {
    "BAAI/bge-small-zh-v1.5": "model.safetensors",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "onnx/model_quint8_avx2.onnx",
}


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare() -> None:
    """仅此阶段联网；下载原配置指定权重及分词/Pooling配置，记录不可变revision。"""
    from huggingface_hub import HfApi, snapshot_download

    manifest = []
    for model, weight in MODELS.items():
        print(f"Preparing {model}", flush=True)
        # 复现时优先使用已公开的revision；首次准备才解析远端main。
        published = ROOT / "docs/experiments/railpolicy_v1/model_manifest.json"
        pinned = {item["model"]: item["revision"] for item in json.loads(published.read_text(encoding="utf-8"))} if published.exists() else {}
        revision = pinned.get(model, "main")
        info = HfApi().model_info(model, revision=revision)
        names = {item.rfilename for item in info.siblings}
        if weight not in names:
            raise ValueError(f"Configured weight unavailable: {model}/{weight}")
        files = sorted(name for name in names if (
            "/" not in name and name.endswith((".json", ".txt", ".model", ".md"))
        ) or name == "1_Pooling/config.json" or name == weight)
        # Windows无symlink权限时，单线程避免hub首次权限探测的并发竞态。
        snapshot = Path(snapshot_download(model, revision=revision, allow_patterns=files, max_workers=1))
        if snapshot.name != info.sha:
            raise ValueError("Model main changed during preparation; retry with a reviewed snapshot")
        # 原工厂保留模型ID而非绝对路径；离线时将缓存main指向已校验的不可变版本。
        ref = snapshot.parent.parent / "refs/main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(info.sha, encoding="utf-8")
        manifest.append({"model": model, "revision": info.sha, "weight": weight,
                         "files_sha256": {name: sha(snapshot / name) for name in files}})
    write(CACHE.parent / "model_manifest.json", manifest)
    print("Model files prepared and hashed; experiments run offline.", flush=True)


def network_guard(allow_hub_metadata: bool) -> list[dict[str, str]]:
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_hub_metadata else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if allow_hub_metadata else "1"
    requests_log: list[dict[str, str]] = []
    context = threading.local()
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def guarded_connect(sock, address):
        if not getattr(context, "hub_metadata", False):
            raise RuntimeError("Network forbidden outside approved Hugging Face metadata reads")
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        if not getattr(context, "hub_metadata", False):
            raise RuntimeError("Network forbidden outside approved Hugging Face metadata reads")
        return connect_ex(sock, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    if allow_hub_metadata:
        import requests
        original_send = requests.Session.send

        def metadata_only(session, request, **kwargs):
            url = urlparse(request.url)
            approved_repo = any(model in url.path for model in MODELS)
            allowed = (url.scheme == "https" and url.hostname == "huggingface.co" and approved_repo
                       and (request.method == "HEAD" or
                            request.method == "GET" and url.path.startswith("/api/models/")))
            if not allowed:
                raise RuntimeError(f"Non-metadata request forbidden: {request.method} {url.hostname}{url.path}")
            requests_log.append({"method": request.method, "path": url.path})
            previous = getattr(context, "hub_metadata", False)
            context.hub_metadata = True
            try:
                return original_send(session, request, **kwargs)
            finally:
                context.hub_metadata = previous

        requests.Session.send = metadata_only
    return requests_log


def verify_models(model_ids: set[str]) -> list[dict]:
    manifest = json.loads((CACHE.parent / "model_manifest.json").read_text(encoding="utf-8"))
    if not model_ids.issubset({item["model"] for item in manifest}):
        raise ValueError("Model manifest differs from fixed experiment models")
    for item in manifest:
        if item["model"] not in model_ids:
            continue
        repo = CACHE / "hub" / ("models--" + item["model"].replace("/", "--"))
        if (repo / "refs/main").read_text(encoding="utf-8").strip() != item["revision"]:
            raise ValueError("Cached main does not resolve to recorded revision")
        for name, digest in item["files_sha256"].items():
            if sha(repo / "snapshots" / item["revision"] / name) != digest:
                raise ValueError(f"Model file changed: {name}")
    return manifest


def check_configs() -> None:
    for original, railway in GROUPS.values():
        base = json.loads((ROOT / "configs" / original).read_text(encoding="utf-8"))
        dev = json.loads((ROOT / "configs" / railway).read_text(encoding="utf-8"))
        expected = dict(base, docs_path="data/railway", eval_path="data/railway/eval_dev.jsonl",
                        evidence_evaluation=True)
        if dev != expected or [dev[k] for k in ("chunk_size", "chunk_overlap", "top_k")] != [420, 80, 3]:
            raise ValueError(f"Uncontrolled experiment configuration: {railway}")


def run(output: Path, allow_hub_metadata: bool = False) -> None:
    check_configs()
    if output.exists():
        raise FileExistsError("Use a new output directory; first-run reports are never overwritten")
    output.mkdir(parents=True)
    network_requests = network_guard(allow_hub_metadata)
    from domainrag.cli import build_retriever, run_benchmark
    from domainrag.io import load_evaluation_cases
    from domainrag.frozen import runtime_fingerprint, text_sha256

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine(), "processor": platform.processor(),
           "device": "cpu", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
           "network_during_evaluation": "HF metadata GET/HEAD only; all other sockets forbidden" if allow_hub_metadata else "disabled by socket guard + HF/Transformers offline",
           "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "runtime": runtime_fingerprint()}
    write(output / "environment.json", env)
    env["preflight"] = {}
    model_manifest = []
    try:
        model_manifest = verify_models({"BAAI/bge-small-zh-v1.5"})
        # 两种模型分别检查；重排不可用时仍完成可运行的B/C组。
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        if encoder.encode(["铁路客运规则加载检查"], show_progress_bar=False).shape != (1, 512):
            raise ValueError("Unexpected BGE embedding shape")
        env["preflight"]["bge"] = {"status": "passed", "device": str(encoder.device)}
        del encoder
    except Exception as exc:
        env["preflight"]["bge"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        verify_models({"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"})
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512,
                               backend="onnx", model_kwargs={"file_name": MODELS["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"]})
        reranker.predict([("退票", "铁路旅客退票规则")], show_progress_bar=False)
        env["preflight"]["reranker"] = {"status": "passed", "providers": getattr(reranker.model, "providers", None)}
        del reranker
    except Exception as exc:
        env["preflight"]["reranker"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write(output / "environment.json", env)
    write(output / "model_manifest.json", model_manifest)
    summaries = {}
    for group, (_, config_name) in GROUPS.items():
        required = [] if group == "A" else ["bge", "reranker"] if group == "D" else ["bge"]
        missing = {name: env["preflight"][name] for name in required if env["preflight"][name]["status"] != "passed"}
        if missing:
            summaries[group] = {"status": "not_run", "reason": missing}
            write(output / f"{group}.json", summaries[group])
            continue
        print(f"Running {group}: {config_name}", flush=True)
        start = time.perf_counter()
        path = output / f"{group}.json"
        try:
            with (output / f"{group}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                run_benchmark(argparse.Namespace(config=str(ROOT / "configs" / config_name), output=str(path)))
            report = json.loads(path.read_text(encoding="utf-8"))
            summaries[group] = {"status": "completed", "summary": report["summary"],
                                "config": config_name, "report_sha256": sha(path),
                                "benchmark_seconds": round(time.perf_counter() - start, 3)}
            # 额外保存固定Top-10候选用于解释重排，正式指标始终来自CLI的Top-3。
            retriever, _ = build_retriever(ROOT / "configs" / config_name)
            diagnostics = []
            for case in load_evaluation_cases(ROOT / "data/railway/eval_dev.jsonl"):
                def items(results):
                    return [{"rank": i, "chunk_id": r.chunk.chunk_id, "source": r.chunk.source,
                             "start": r.chunk.start, "end": r.chunk.end, "score": round(r.score, 6)}
                            for i, r in enumerate(results, 1)]
                ranked = items(retriever.search(case.question, top_k=10))
                actual = next(row["retrieved"] for row in report["cases"] if row["id"] == case.case_id)
                if ranked[:3] != [{k: r[k] for k in ranked[0]} for r in actual]:
                    raise ValueError(f"Top-10 diagnostic changed official Top-3: {case.case_id}")
                row = {"id": case.case_id, "ranked_top10": ranked}
                if group == "D":
                    row["before_reranking_top10"] = items(retriever.retriever.search(case.question, top_k=10))
                diagnostics.append(row)
            write(output / f"{group}.candidates.json", diagnostics)
            print(json.dumps(summaries[group]["summary"], ensure_ascii=False), flush=True)
        except Exception as exc:
            summaries[group] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            write(output / f"{group}.error.json", summaries[group])
            print(summaries[group], flush=True)
    # 确认元数据访问没有替换固定缓存权重/版本。
    verify_models(set(MODELS))
    write(output / "network_metadata_requests.json", network_requests)
    write(output / "comparison.json", {"experiment": "RailPolicy-RAG V1", "groups": summaries,
          "eval_sha256": text_sha256(ROOT / "data/railway/eval_dev.jsonl"),
          "environment": "environment.json", "models": "model_manifest.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="download configured model files only")
    parser.add_argument("--output", default="results/railpolicy_v1/run_v1")
    parser.add_argument("--allow-hub-metadata", action="store_true", help="allow only official HF GET metadata / HEAD, for ONNX loaders requiring repo listing")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run((ROOT / args.output).resolve(), args.allow_hub_metadata)
