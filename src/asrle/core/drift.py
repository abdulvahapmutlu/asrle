from __future__ import annotations

from asrle.types import DriftPoint, DriftReport, Segment


def detect_timestamp_drift(segments: list[Segment]) -> DriftReport:
    """
    Drift detection without word-level forced alignment:
    - checks monotonic segment timings, overlaps, long gaps, and suspicious pacing.
    - reports "drift-like" symptoms common in chunked/streaming systems.
    """
    if not segments:
        return DriftReport(drift_points=[], max_abs_drift_s=0.0, suspicious=False, reasons=[])

    reasons: list[str] = []
    drift_points: list[DriftPoint] = []
    max_abs = 0.0

    last_end = segments[0].start_s
    for s in segments:
        if s.start_s < last_end - 0.05:
            reasons.append(f"Overlap detected: seg starts at {s.start_s:.2f}s < prev end {last_end:.2f}s")
        gap = s.start_s - last_end
        if gap > 1.0:
            reasons.append(f"Large gap {gap:.2f}s before segment starting at {s.start_s:.2f}s")

        # "Drift point" heuristic: if segments stop being contiguous, treat it as drift.
        drift = gap
        if abs(drift) > max_abs:
            max_abs = abs(drift)
        drift_points.append(DriftPoint(t_s=s.start_s, drift_s=float(drift), note=None))

        last_end = max(last_end, s.end_s)

    suspicious = bool(reasons) and max_abs > 1.0
    return DriftReport(
        drift_points=drift_points,
        max_abs_drift_s=float(max_abs),
        suspicious=suspicious,
        reasons=reasons,
    )
