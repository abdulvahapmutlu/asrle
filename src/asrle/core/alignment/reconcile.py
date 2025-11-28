from __future__ import annotations

from typing import Iterable

from asrle.types import WordStamp


def reconcile_words(words: Iterable[WordStamp], min_dur_s: float = 0.02) -> list[WordStamp]:
    """
    Enforces:
    - non-decreasing timestamps
    - end >= start
    - small minimum word duration so plots don't collapse
    """
    out: list[WordStamp] = []
    last_end = 0.0
    for w in words:
        s = max(0.0, float(w.start_s))
        e = max(s + min_dur_s, float(w.end_s))
        s = max(s, last_end)
        e = max(e, s + min_dur_s)
        out.append(
            WordStamp(
                word=w.word,
                start_s=s,
                end_s=e,
                confidence=w.confidence,
                source=w.source,
            )
        )
        last_end = e
    return out
