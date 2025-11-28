from __future__ import annotations

from dataclasses import asdict
from typing import Any

from asrle.types import ErrorMoment, MomentsReport, WordAttributionReport


def _parse_bin_label(label: str) -> tuple[float, float]:
    # expected "0-1s" or "10-11s"
    s = label.strip().lower()
    s = s[:-1] if s.endswith("s") else s
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad bin label: {label}")
    return float(parts[0]), float(parts[1])


def _score(counts: dict[str, int]) -> float:
    # substitutions usually most meaningful
    sub = int(counts.get("sub", 0))
    ins = int(counts.get("ins", 0))
    dele = int(counts.get("del", 0))
    return float(2.0 * sub + 1.0 * ins + 1.0 * dele)


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) < min(a1, b1)


def build_error_moments(
    wa: WordAttributionReport,
    *,
    top_k: int = 3,
    bin_s: float = 1.0,
    pad_s: float = 0.6,
    min_separation_s: float = 0.8,
    max_sample_events: int = 8,
) -> MomentsReport:
    """
    Returns timestamp-only "clip suggestions" where word-level errors spike.

    - Uses wa.time_bins (1s bins by default)
    - Score = 2*sub + ins + del
    - Greedy selection with non-overlap and minimum separation
    """
    if not wa.time_bins:
        return MomentsReport(bin_s=bin_s, pad_s=pad_s, moments=[])

    bins: list[dict[str, Any]] = []
    for label, counts in wa.time_bins.items():
        try:
            b0, b1 = _parse_bin_label(label)
        except Exception:
            continue
        c = {
            "hit": int(counts.get("hit", 0)),
            "sub": int(counts.get("sub", 0)),
            "ins": int(counts.get("ins", 0)),
            "del": int(counts.get("del", 0)),
        }
        sc = _score(c)
        if sc <= 0:
            continue
        bins.append({"label": label, "b0": b0, "b1": b1, "counts": c, "score": sc})

    bins.sort(key=lambda x: x["score"], reverse=True)

    chosen: list[ErrorMoment] = []
    chosen_ranges: list[tuple[float, float]] = []

    for cand in bins:
        if len(chosen) >= top_k:
            break

        b0, b1 = float(cand["b0"]), float(cand["b1"])
        w0 = max(0.0, b0 - pad_s)
        w1 = b1 + pad_s

        # ensure separation: reject if too close/overlaps chosen windows
        ok = True
        for (c0, c1) in chosen_ranges:
            if _overlaps(w0 - min_separation_s, w1 + min_separation_s, c0, c1):
                ok = False
                break
        if not ok:
            continue

        # sample events within window (non-hits first)
        sample = []
        for e in wa.events:
            if float(e.start_s) < w0 or float(e.start_s) > w1:
                continue
            if e.op == "hit":
                continue
            sample.append(
                {
                    "op": e.op,
                    "t": float(e.start_s),
                    "ref": e.ref_word,
                    "hyp": e.hyp_word,
                    "conf": e.confidence,
                }
            )
            if len(sample) >= max_sample_events:
                break

        chosen.append(
            ErrorMoment(
                rank=len(chosen) + 1,
                start_s=float(w0),
                end_s=float(w1),
                window_label=str(cand["label"]),
                counts=dict(cand["counts"]),
                score=float(cand["score"]),
                sample_events=sample,
            )
        )
        chosen_ranges.append((float(w0), float(w1)))

    return MomentsReport(bin_s=bin_s, pad_s=pad_s, moments=chosen)
