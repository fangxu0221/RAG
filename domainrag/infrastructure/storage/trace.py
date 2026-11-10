"""每个请求独占的 JSONL 文件。"""

import json
from pathlib import Path


class JsonlTrace:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", encoding="utf-8")

    def write(self, event: dict) -> None:
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()
