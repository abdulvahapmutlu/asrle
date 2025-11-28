from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-\.]+")


def slugify(name: str, max_len: int = 64) -> str:
    s = (name or "").strip()
    s = SAFE_NAME_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "run"
    return s[:max_len]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def now_run_id(prefix: str = "run") -> str:
    # YYYYmmdd_HHMMSS + short monotonic tail
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tail = str(int(time.time() * 1000) % 100000).zfill(5)
    return f"{prefix}_{ts}_{tail}"


def write_json(path: str, obj: Any) -> str:
    ensure_dir(os.path.dirname(path))

    def _default(o):
        if is_dataclass(o):
            return asdict(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)
    return path


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_uploaded_file(uploaded_file, out_path: str) -> str:
    """
    Streamlit UploadedFile -> save on disk.
    """
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def list_run_dirs(base_dir: str) -> list[str]:
    """
    Returns run directories that include report.json.
    """
    if not os.path.isdir(base_dir):
        return []
    out = []
    for name in sorted(os.listdir(base_dir), reverse=True):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "report.json")):
            out.append(p)
        elif os.path.exists(os.path.join(p, "dataset_summary.json")):
            out.append(p)
    return out


def human_relpath(path: str, base: str) -> str:
    try:
        return os.path.relpath(path, base)
    except Exception:
        return path


def zip_dir(src_dir: str, out_zip_path: str | None = None) -> str:
    """
    Zip a directory. If out_zip_path None, create in temp.
    """
    src_dir = os.path.abspath(src_dir)
    if out_zip_path is None:
        td = tempfile.mkdtemp(prefix="asrle_zip_")
        out_zip_path = os.path.join(td, os.path.basename(src_dir) + ".zip")

    ensure_dir(os.path.dirname(out_zip_path))
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(src_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                arc = os.path.relpath(fp, src_dir)
                z.write(fp, arcname=arc)

    return out_zip_path


def safe_join(base: str, *parts: str) -> str:
    """
    Prevent directory traversal: resolves and ensures inside base.
    """
    base_abs = os.path.abspath(base)
    p = os.path.abspath(os.path.join(base_abs, *parts))
    if not p.startswith(base_abs):
        raise ValueError("Unsafe path outside base directory.")
    return p


def try_read_text(path: str, limit: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read(limit + 1)
        return s[:limit]
    except Exception:
        return ""
