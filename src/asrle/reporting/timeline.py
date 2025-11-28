from __future__ import annotations

from typing import Any

from asrle.types import WordAttributionReport


def build_timeline_series(wa: WordAttributionReport) -> list[dict[str, Any]]:
    """
    Converts wa.time_bins dict -> sorted list:
    [
      {"start_s":0.0,"end_s":1.0,"label":"0-1s","hit":..,"sub":..,"ins":..,"del":..,"score":..},
      ...
    ]
    """
    out: list[dict[str, Any]] = []

    for label, counts in (wa.time_bins or {}).items():
        s = label.strip().lower()
        s = s[:-1] if s.endswith("s") else s
        parts = s.split("-")
        if len(parts) != 2:
            continue
        try:
            b0 = float(parts[0])
            b1 = float(parts[1])
        except Exception:
            continue

        hit = int(counts.get("hit", 0))
        sub = int(counts.get("sub", 0))
        ins = int(counts.get("ins", 0))
        dele = int(counts.get("del", 0))
        score = float(2.0 * sub + 1.0 * ins + 1.0 * dele)

        out.append(
            {
                "start_s": b0,
                "end_s": b1,
                "label": label,
                "hit": hit,
                "sub": sub,
                "ins": ins,
                "del": dele,
                "score": score,
            }
        )

    out.sort(key=lambda x: (x["start_s"], x["end_s"]))
    return out
