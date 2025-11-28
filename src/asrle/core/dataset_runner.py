from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from asrle.backends.registry import BackendRegistry
from asrle.core.engine import Engine
from asrle.types import DatasetItemResult, DatasetSummary
from asrle.utils.io import ensure_dir, read_text


@dataclass(frozen=True)
class DatasetRunnerConfig:
    manifest_path: str
    out_dir: str = "runs/dataset"
    backend: str = "hf-whisper"
    backend_params: dict[str, Any] = None  # type: ignore[assignment]
    repeats: int = 1
    per_item_dir: bool = True

    # word alignment / attribution
    word_align: bool = False
    word_attribution: bool = True
    aligner_model: str = "facebook/wav2vec2-base-960h"
    aligner_device: str = "cpu"

    # streaming
    streaming_enabled: bool = False
    chunk_ms: int = 800
    overlap_ms: int = 160
    right_context_ms: int = 0


def _float_or_none(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _bool_or_none(x: str | None) -> bool | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def read_manifest(manifest_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: v for k, v in r.items()})
    if not rows:
        raise ValueError("Manifest is empty.")
    if "audio_path" not in rows[0]:
        raise ValueError("Manifest must contain 'audio_path' column.")
    return rows


def run_dataset(cfg: DatasetRunnerConfig) -> DatasetSummary:
    ensure_dir(cfg.out_dir)
    rows = read_manifest(cfg.manifest_path)

    reg = BackendRegistry()
    engine = Engine(registry=reg)

    backend_params = cfg.backend_params or {}

    items: list[DatasetItemResult] = []
    ok = 0
    failed = 0

    for i, row in enumerate(tqdm(rows, desc="ASR-LE dataset")):
        audio = row.get("audio_path", "")
        if not audio:
            items.append(
                DatasetItemResult(
                    audio_path="",
                    ok=False,
                    backend=cfg.backend,
                    duration_s=None,
                    wer=None,
                    p95_latency_s=None,
                    first_word_latency_s=None,
                    meta=row,
                    error="Missing audio_path",
                )
            )
            failed += 1
            continue

        ref_text = row.get("ref_text")
        ref_path = row.get("ref_path")
        reference = None
        if ref_text:
            reference = str(ref_text)
        elif ref_path:
            try:
                reference = read_text(str(ref_path))
            except Exception:
                reference = None

        item_out = os.path.join(cfg.out_dir, f"item_{i:05d}") if cfg.per_item_dir else cfg.out_dir
        ensure_dir(item_out)

        try:
            report = engine.analyze(
                audio_path=str(audio),
                backend_name=cfg.backend,
                backend_params=dict(backend_params),
                reference_text=reference,
                repeats=cfg.repeats,
                enable_profiling=True,
                compute_wer_flag=True,
                compute_attribution_flag=True,
                compute_drift_flag=True,
                enable_suggestions_flag=True,
                word_align=cfg.word_align,
                word_attribution=cfg.word_attribution,
                aligner_model=cfg.aligner_model,
                aligner_device=cfg.aligner_device,
                streaming_enabled=cfg.streaming_enabled,
                chunk_ms=cfg.chunk_ms,
                overlap_ms=cfg.overlap_ms,
                right_context_ms=cfg.right_context_ms,
                out_dir=item_out,
            )

            wer = report.wer.wer if report.wer else None
            p95 = report.latency.percentiles_s.get("p95") if report.latency else None
            fwlat = None
            if report.streaming:
                # use p95 estimator if repeats>1 else single sample
                fwlat = (
                    report.streaming.first_word_latency_percentiles_s.get("p95")
                    if report.streaming.first_word_latency_percentiles_s
                    else report.streaming.first_word_latency_s
                )

            meta = dict(row)
            meta["snr_db"] = _float_or_none(row.get("snr_db"))
            meta["noise_type"] = row.get("noise_type") or None
            meta["far_field"] = _bool_or_none(row.get("far_field"))

            items.append(
                DatasetItemResult(
                    audio_path=str(audio),
                    ok=True,
                    backend=cfg.backend,
                    duration_s=report.audio_duration_s,
                    wer=wer,
                    p95_latency_s=p95,
                    first_word_latency_s=fwlat,
                    meta=meta,
                    out_dir=item_out,
                )
            )
            ok += 1
        except Exception as e:
            items.append(
                DatasetItemResult(
                    audio_path=str(audio),
                    ok=False,
                    backend=cfg.backend,
                    duration_s=None,
                    wer=None,
                    p95_latency_s=None,
                    first_word_latency_s=None,
                    meta=dict(row),
                    error=str(e),
                    out_dir=item_out,
                )
            )
            failed += 1

    return DatasetSummary(
        backend=cfg.backend,
        backend_params=dict(backend_params),
        n=len(items),
        ok=ok,
        failed=failed,
        wer_mean=None,
        wer_p50=None,
        wer_p90=None,
        latency_p50_s=None,
        latency_p95_s=None,
        fwlat_p50_s=None,
        fwlat_p95_s=None,
        slices={},
        items=items,
    )
