from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from asrle.types import Segment, Transcript, WordAttributionReport, WordErrorEvent, WordStamp
from asrle.utils.text import normalize_text


def _extract_words_from_meta(transcript: Transcript) -> list[WordStamp]:
    words: list[WordStamp] = []
    meta_words = transcript.meta.get("words")
    if isinstance(meta_words, list):
        for w in meta_words:
            try:
                words.append(
                    WordStamp(
                        word=str(w.get("word", "")).strip(),
                        start_s=float(w.get("start_s", 0.0)),
                        end_s=float(w.get("end_s", 0.0)),
                        confidence=(float(w["confidence"]) if w.get("confidence") is not None else None),
                        source=str(w.get("source", "backend")),
                    )
                )
            except Exception:
                continue
    return words


def _approx_words_from_segments(segments: list[Segment]) -> list[WordStamp]:
    """
    Approximate word timestamps by distributing segment duration across segment words.
    Used when backend does not provide word-level timestamps.
    """
    out: list[WordStamp] = []
    last_end = 0.0
    for seg in segments:
        txt = seg.text.strip()
        words = [w for w in normalize_text(txt).split() if w]
        if not words:
            continue

        s0 = float(seg.start_s)
        e0 = float(seg.end_s)
        if e0 <= s0:
            # no timing info; fall back to monotonic fake timing
            s0 = last_end
            e0 = last_end + max(0.2, 0.12 * len(words))

        dur = max(0.02, e0 - s0)
        step = dur / max(1, len(words))
        for i, w in enumerate(words):
            s = s0 + i * step
            e = s0 + (i + 1) * step
            out.append(WordStamp(word=w, start_s=s, end_s=e, confidence=None, source="heuristic"))
        last_end = max(last_end, e0)
    return out


def get_hyp_words(transcript: Transcript) -> list[WordStamp]:
    words = _extract_words_from_meta(transcript)
    if words:
        return words
    return _approx_words_from_segments(transcript.segments)


def _dp_align_ops(ref: list[str], hyp: list[str]) -> list[tuple[Literal["hit", "sub", "ins", "del"], int | None, int | None]]:
    """
    Word-level alignment via edit distance DP + backtrace.
    Returns a list of ops with indices into ref/hyp.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = (i - 1, 0, "del")
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = (0, j - 1, "ins")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            # del, ins, sub/hit
            cand = [
                (dp[i - 1][j] + 1, (i - 1, j, "del")),
                (dp[i][j - 1] + 1, (i, j - 1, "ins")),
                (dp[i - 1][j - 1] + cost, (i - 1, j - 1, "sub" if cost else "hit")),
            ]
            best = min(cand, key=lambda x: x[0])
            dp[i][j] = best[0]
            bt[i][j] = best[1]

    ops: list[tuple[Literal["hit", "sub", "ins", "del"], int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        prev = bt[i][j]
        if prev is None:
            break
        pi, pj, op = prev
        if op in ("hit", "sub"):
            ops.append((op, i - 1, j - 1))
        elif op == "del":
            ops.append(("del", i - 1, None))
        else:  # ins
            ops.append(("ins", None, j - 1))
        i, j = pi, pj
    ops.reverse()
    return ops


def _bin_key(t: float, bin_s: float) -> str:
    b0 = int(t // bin_s)
    return f"{b0*bin_s:.0f}-{(b0+1)*bin_s:.0f}s"


def build_word_attribution(ref_words: list[WordStamp], hyp_words: list[WordStamp], bin_s: float = 1.0) -> WordAttributionReport:
    ref_norm = [normalize_text(w.word) for w in ref_words]
    hyp_norm = [normalize_text(w.word) for w in hyp_words]

    ops = _dp_align_ops(ref_norm, hyp_norm)

    events: list[WordErrorEvent] = []
    time_bins: dict[str, dict[str, int]] = {}

    def bump(t: float, op: str) -> None:
        k = _bin_key(t, bin_s)
        if k not in time_bins:
            time_bins[k] = {"hit": 0, "sub": 0, "ins": 0, "del": 0}
        time_bins[k][op] = int(time_bins[k].get(op, 0)) + 1

    for op, iref, ihyp in ops:
        rw = ref_words[iref] if iref is not None else None
        hw = hyp_words[ihyp] if ihyp is not None else None

        # choose event time window: prefer hyp time, else ref time, else 0
        if hw is not None:
            s, e = hw.start_s, hw.end_s
            conf = hw.confidence
        elif rw is not None:
            s, e = rw.start_s, rw.end_s
            conf = rw.confidence
        else:
            s, e, conf = 0.0, 0.01, None

        # highlight substitutions by time (use start)
        bump(float(s), op)

        events.append(
            WordErrorEvent(
                op=op,
                start_s=float(s),
                end_s=float(max(e, s + 0.01)),
                ref_word=(rw.word if rw else None),
                hyp_word=(hw.word if hw else None),
                ref_start_s=(rw.start_s if rw else None),
                ref_end_s=(rw.end_s if rw else None),
                hyp_start_s=(hw.start_s if hw else None),
                hyp_end_s=(hw.end_s if hw else None),
                confidence=conf,
                note=None,
            )
        )

    # top substitution windows
    tops = []
    for k, v in time_bins.items():
        tops.append({"window": k, "sub": v.get("sub", 0), "ins": v.get("ins", 0), "del": v.get("del", 0)})
    tops.sort(key=lambda x: (x["sub"], x["ins"] + x["del"]), reverse=True)

    return WordAttributionReport(
        events=events,
        time_bins=time_bins,
        top_substitution_windows=tops[:10],
    )
