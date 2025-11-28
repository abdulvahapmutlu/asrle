from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


def write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
