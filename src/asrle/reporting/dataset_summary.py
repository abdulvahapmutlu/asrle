from __future__ import annotations

import os
from collections import defaultdict
from statistics import mean

from asrle.types import DatasetItemResult, DatasetSummary
from asrle.utils.io import ensure_dir, write_json


def _pct(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty")
    v = sorted(values)
    idx = int(round(q * (len(v) - 1)))
    return float(v[max(0, min(len(v) - 1, idx))])


def _safe_mean(vals: list[float]) -> float | None:
    return float(mean(vals)) if vals else None


def compute_dataset_summary(summary: DatasetSummary) -> DatasetSummary:
    wers = [x.wer for x in summary.items if x.ok and x.wer is not None]
    lats = [x.p95_latency_s for x in summary.items if x.ok and x.p95_latency_s is not None]
    fwls = [x.first_word_latency_s for x in summary.items if x.ok and x.first_word_latency_s is not None]

    wer_vals = [float(x) for x in wers]
    lat_vals = [float(x) for x in lats]
    fwl_vals = [float(x) for x in fwls]

    slices: dict[str, object] = {}

    # by noise_type
    by_noise: dict[str, list[DatasetItemResult]] = defaultdict(list)
    for it in summary.items:
        k = str(it.meta.get("noise_type") or "unknown")
        by_noise[k].append(it)

    noise_stats = {}
    for k, items in by_noise.items():
        w = [x.wer for x in items if x.ok and x.wer is not None]
        f = [x.first_word_latency_s for x in items if x.ok and x.first_word_latency_s is not None]
        noise_stats[k] = {
            "n": len(items),
            "ok": sum(1 for x in items if x.ok),
            "wer_mean": _safe_mean([float(x) for x in w]),
            "fwlat_mean_s": _safe_mean([float(x) for x in f]),
        }
    slices["noise_type"] = noise_stats

    # by far_field
    by_far = {"true": [], "false": [], "unknown": []}
    for it in summary.items:
        ff = it.meta.get("far_field")
        if ff is True:
            by_far["true"].append(it)
        elif ff is False:
            by_far["false"].append(it)
        else:
            by_far["unknown"].append(it)

    far_stats = {}
    for k, items in by_far.items():
        w = [x.wer for x in items if x.ok and x.wer is not None]
        f = [x.first_word_latency_s for x in items if x.ok and x.first_word_latency_s is not None]
        far_stats[k] = {
            "n": len(items),
            "ok": sum(1 for x in items if x.ok),
            "wer_mean": _safe_mean([float(x) for x in w]),
            "fwlat_mean_s": _safe_mean([float(x) for x in f]),
        }
    slices["far_field"] = far_stats

    # SNR bins
    snr_bins = {"<10": [], "10-20": [], "20-30": [], ">=30": [], "unknown": []}
    for it in summary.items:
        snr = it.meta.get("snr_db")
        if snr is None:
            snr_bins["unknown"].append(it)
        else:
            s = float(snr)
            if s < 10:
                snr_bins["<10"].append(it)
            elif s < 20:
                snr_bins["10-20"].append(it)
            elif s < 30:
                snr_bins["20-30"].append(it)
            else:
                snr_bins[">=30"].append(it)

    snr_stats = {}
    for k, items in snr_bins.items():
        w = [x.wer for x in items if x.ok and x.wer is not None]
        f = [x.first_word_latency_s for x in items if x.ok and x.first_word_latency_s is not None]
        snr_stats[k] = {
            "n": len(items),
            "ok": sum(1 for x in items if x.ok),
            "wer_mean": _safe_mean([float(x) for x in w]),
            "fwlat_mean_s": _safe_mean([float(x) for x in f]),
        }
    slices["snr_db"] = snr_stats

    return DatasetSummary(
        backend=summary.backend,
        backend_params=summary.backend_params,
        n=summary.n,
        ok=summary.ok,
        failed=summary.failed,
        wer_mean=_safe_mean(wer_vals),
        wer_p50=_pct(wer_vals, 0.50) if wer_vals else None,
        wer_p90=_pct(wer_vals, 0.90) if wer_vals else None,
        latency_p50_s=_pct(lat_vals, 0.50) if lat_vals else None,
        latency_p95_s=_pct(lat_vals, 0.95) if lat_vals else None,
        fwlat_p50_s=_pct(fwl_vals, 0.50) if fwl_vals else None,
        fwlat_p95_s=_pct(fwl_vals, 0.95) if fwl_vals else None,
        slices=slices,
        items=summary.items,
    )


def write_dataset_summary(summary: DatasetSummary, out_dir: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "dataset_summary.json")
    write_json(path, summary)
    return path
