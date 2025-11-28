from __future__ import annotations

import csv
import os
import tempfile
from typing import Any, Callable

import numpy as np

from asrle.backends.registry import BackendRegistry
from asrle.contracts.backend_contracts import validate_backend_contract
from asrle.contracts.streaming_contracts import validate_streaming_contract
from asrle.core.dataset_runner import read_manifest
from asrle.core.engine import Engine
from asrle.reporting.dataset_summary import compute_dataset_summary, write_dataset_summary
from asrle.utils.audio import write_wav
from asrle.utils.io import ensure_dir

ProgressFn = Callable[[float, str], None]


def _list_backend_names(reg: BackendRegistry) -> list[str]:
    # best-effort introspection to support community registries
    for attr in ("available", "list_backends", "list", "names"):
        if hasattr(reg, attr) and callable(getattr(reg, attr)):
            try:
                names = list(getattr(reg, attr)())
                if names:
                    return sorted(set(map(str, names)))
            except Exception:
                pass

    for attr in ("_builders", "_registry", "builders", "backends", "FACTORIES", "REGISTRY"):
        d = getattr(reg, attr, None)
        if isinstance(d, dict) and d:
            return sorted(set(map(str, d.keys())))

    # fallback
    return ["dummy", "hf-whisper", "faster-whisper"]


def get_engine_and_registry() -> tuple[Engine, BackendRegistry, list[str]]:
    reg = BackendRegistry()
    eng = Engine(registry=reg)
    return eng, reg, _list_backend_names(reg)


def run_single_analysis(
    *,
    audio_path: str,
    out_dir: str,
    backend_name: str,
    backend_params: dict[str, Any],
    reference_text: str | None,
    repeats: int,
    word_align: bool,
    aligner_model: str,
    aligner_device: str,
    word_attribution: bool,
    streaming_enabled: bool,
    chunk_ms: int,
    overlap_ms: int,
    right_context_ms: int,
    torch_trace: bool,
    enable_profiling: bool,
) -> str:
    ensure_dir(out_dir)
    eng, _reg, _names = get_engine_and_registry()

    _ = eng.analyze(
        audio_path=audio_path,
        backend_name=backend_name,
        backend_params=backend_params,
        reference_text=reference_text,
        repeats=max(1, int(repeats)),
        enable_profiling=bool(enable_profiling),
        compute_wer_flag=True,
        compute_attribution_flag=True,
        compute_drift_flag=True,
        enable_suggestions_flag=True,
        word_align=bool(word_align),
        aligner_model=aligner_model,
        aligner_device=aligner_device,
        word_attribution=bool(word_attribution),
        streaming_enabled=bool(streaming_enabled),
        chunk_ms=int(chunk_ms),
        overlap_ms=int(overlap_ms),
        right_context_ms=int(right_context_ms),
        out_dir=out_dir,
        torch_trace=bool(torch_trace),
    )

    return out_dir


def run_dataset_web(
    *,
    manifest_path: str,
    out_dir: str,
    backend_name: str,
    backend_params: dict[str, Any],
    repeats: int,
    word_align: bool,
    aligner_model: str,
    aligner_device: str,
    word_attribution: bool,
    streaming_enabled: bool,
    chunk_ms: int,
    overlap_ms: int,
    right_context_ms: int,
    progress: ProgressFn | None = None,
) -> str:
    """
    Web-friendly dataset runner (progress callback).
    Compatible with the manifest format used by read_manifest().
    Writes per-item outputs under out_dir/items/item_XXXXX and dataset_summary.json at out_dir.
    """
    ensure_dir(out_dir)
    items_dir = os.path.join(out_dir, "items")
    ensure_dir(items_dir)

    eng, _reg, _names = get_engine_and_registry()

    rows = read_manifest(manifest_path)
    n = len(rows)

    ok = 0
    failed = 0
    item_results: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        frac = 0.0 if n == 0 else float(i) / float(n)
        if progress:
            progress(frac, f"Processing item {i+1}/{n}")

        audio = str(row.get("audio_path") or "").strip()
        if not audio:
            failed += 1
            item_results.append({"audio_path": "", "ok": False, "error": "Missing audio_path", "meta": row})
            continue

        ref = None
        if row.get("ref_text"):
            ref = str(row.get("ref_text"))
        elif row.get("ref_path"):
            try:
                with open(str(row.get("ref_path")), "r", encoding="utf-8") as f:
                    ref = f.read()
            except Exception:
                ref = None

        item_out = os.path.join(items_dir, f"item_{i:05d}")
        ensure_dir(item_out)

        try:
            report = eng.analyze(
                audio_path=audio,
                backend_name=backend_name,
                backend_params=dict(backend_params),
                reference_text=ref,
                repeats=max(1, int(repeats)),
                enable_profiling=True,
                compute_wer_flag=True,
                compute_attribution_flag=True,
                compute_drift_flag=True,
                enable_suggestions_flag=True,
                word_align=bool(word_align),
                aligner_model=aligner_model,
                aligner_device=aligner_device,
                word_attribution=bool(word_attribution),
                streaming_enabled=bool(streaming_enabled),
                chunk_ms=int(chunk_ms),
                overlap_ms=int(overlap_ms),
                right_context_ms=int(right_context_ms),
                out_dir=item_out,
            )

            wer = report.wer.wer if report.wer else None
            p95 = report.latency.percentiles_s.get("p95") if report.latency else None
            fwlat = None
            if report.streaming:
                fwlat = (
                    report.streaming.first_word_latency_percentiles_s.get("p95")
                    if report.streaming.first_word_latency_percentiles_s
                    else report.streaming.first_word_latency_s
                )

            ok += 1
            item_results.append(
                {
                    "audio_path": audio,
                    "ok": True,
                    "backend": backend_name,
                    "duration_s": report.audio_duration_s,
                    "wer": wer,
                    "p95_latency_s": p95,
                    "first_word_latency_s": fwlat,
                    "meta": dict(row),
                    "out_dir": item_out,
                }
            )
        except Exception as e:
            failed += 1
            item_results.append(
                {
                    "audio_path": audio,
                    "ok": False,
                    "backend": backend_name,
                    "error": str(e),
                    "meta": dict(row),
                    "out_dir": item_out,
                }
            )

    # Build a DatasetSummary-like dict compatible with compute_dataset_summary by using asrle types layout
    # We’ll just write a lightweight JSON and also a proper dataset_summary.json via helper.
    # If compute_dataset_summary expects DatasetSummary dataclass, it will still work if objects were created upstream.
    # Here we use the provided writer which expects DatasetSummary, so we compute summary ourselves too.
    # To keep it robust, we write a simple summary.json + a "dataset_summary.json" with the same schema we display.

    wers = [x["wer"] for x in item_results if x.get("ok") and x.get("wer") is not None]
    lats = [x["p95_latency_s"] for x in item_results if x.get("ok") and x.get("p95_latency_s") is not None]
    fwls = [x["first_word_latency_s"] for x in item_results if x.get("ok") and x.get("first_word_latency_s") is not None]

    def pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        v = sorted(float(x) for x in vals)
        idx = int(round(q * (len(v) - 1)))
        idx = max(0, min(len(v) - 1, idx))
        return float(v[idx])

    def mean(vals: list[float]) -> float | None:
        if not vals:
            return None
        return float(sum(float(x) for x in vals) / len(vals))

    summary_json = {
        "backend": backend_name,
        "backend_params": dict(backend_params),
        "n": len(item_results),
        "ok": ok,
        "failed": failed,
        "wer_mean": mean([float(x) for x in wers if x is not None]),
        "wer_p50": pct([float(x) for x in wers if x is not None], 0.50),
        "wer_p90": pct([float(x) for x in wers if x is not None], 0.90),
        "latency_p50_s": pct([float(x) for x in lats if x is not None], 0.50),
        "latency_p95_s": pct([float(x) for x in lats if x is not None], 0.95),
        "fwlat_p50_s": pct([float(x) for x in fwls if x is not None], 0.50),
        "fwlat_p95_s": pct([float(x) for x in fwls if x is not None], 0.95),
        "slices": {},  # filled by compute_dataset_summary if you run CLI pipeline
        "items": item_results,
    }

    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    if progress:
        progress(1.0, "Done.")
    return out_dir


def validate_backend_web(backend_name: str) -> dict[str, Any]:
    eng, reg, _names = get_engine_and_registry()
    _ = eng

    b = reg.create(backend_name)

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "silence.wav")
        write_wav(wav, np.zeros((16000,), dtype=np.float32), 16000)

        ok1, issues1 = validate_backend_contract(b, wav)
        ok2, issues2 = validate_streaming_contract(b)

    return {
        "backend": backend_name,
        "transcribe_contract_ok": bool(ok1),
        "transcribe_issues": list(issues1),
        "streaming_contract_ok": bool(ok2),
        "streaming_issues": list(issues2),
    }
