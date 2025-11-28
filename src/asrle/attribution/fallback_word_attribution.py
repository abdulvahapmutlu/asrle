from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re

_word_re = re.compile(r"[^a-z0-9' ]+")

def _norm_word(w: str) -> str:
    w = (w or "").lower().replace("\ufeff", "").replace("-", " ")
    w = _word_re.sub(" ", w)
    w = re.sub(r"\s+", " ", w).strip()
    return w

def _norm_words_from_text(text: str) -> List[str]:
    return [t for t in _norm_word(text).split(" ") if t]

def _norm_words_from_hyp(hyp_words: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for x in hyp_words:
        out.append(_norm_word(str(x.get("word", ""))))
    return out

def _lev_ops(ref: List[str], hyp: List[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
    # returns list of (op, i_ref, i_hyp) where op in {hit, sub, ins, del}
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"
    bt[0][0] = "done"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            best_cost = dp[i - 1][j - 1] + cost
            best_op = "hit" if cost == 0 else "sub"

            c_del = dp[i - 1][j] + 1
            if c_del < best_cost:
                best_cost, best_op = c_del, "del"

            c_ins = dp[i][j - 1] + 1
            if c_ins < best_cost:
                best_cost, best_op = c_ins, "ins"

            dp[i][j] = best_cost
            bt[i][j] = best_op

    ops: List[Tuple[str, Optional[int], Optional[int]]] = []
    i, j = n, m
    while not (i == 0 and j == 0):
        op = bt[i][j]
        if op in ("hit", "sub"):
            ops.append((op, i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "del":
            ops.append(("del", i - 1, None))
            i -= 1
        elif op == "ins":
            ops.append(("ins", None, j - 1))
            j -= 1
        else:
            break
    ops.reverse()
    return ops

def build_word_attribution_fallback(
    reference_text: str,
    hyp_words: List[Dict[str, Any]],
    audio_duration_s: float,
    bin_s: float = 1.0,
) -> Dict[str, Any]:
    ref = _norm_words_from_text(reference_text)
    hyp = _norm_words_from_hyp(hyp_words)

    ops = _lev_ops(ref, hyp)

    events: List[Dict[str, Any]] = []
    for op, i_ref, i_hyp in ops:
        ref_w = ref[i_ref] if i_ref is not None and i_ref < len(ref) else None
        hyp_w = hyp_words[i_hyp]["word"] if i_hyp is not None and i_hyp < len(hyp_words) else None

        # timestamp strategy: use hyp timestamps when available
        if i_hyp is not None and i_hyp < len(hyp_words):
            hs = float(hyp_words[i_hyp].get("start_s", 0.0))
            he = float(hyp_words[i_hyp].get("end_s", hs))
            conf = hyp_words[i_hyp].get("confidence", None)
        else:
            # deletion: place it near neighboring hyp word times (safe approximate)
            hs, he, conf = 0.0, 0.0, None

        events.append({
            "op": op if op != "hit" else "hit",
            "start_s": hs,
            "end_s": he,
            "ref_word": ref_w,
            "hyp_word": hyp_w,
            "confidence": conf,
            "note": "fallback_no_ref_timestamps",
        })

    # bin events
    def bin_label(t0: float, t1: float) -> str:
        a = int(t0 // bin_s)
        b = int(t1 // bin_s)
        # pick the bin where the event starts
        return f"{a}-{a+1}s" if bin_s == 1.0 else f"{a*bin_s:.2f}-{(a+1)*bin_s:.2f}s"

    time_bins: Dict[str, Dict[str, int]] = {}
    for e in events:
        if e["op"] == "hit":
            continue
        lbl = bin_label(float(e["start_s"]), float(e["end_s"]))
        time_bins.setdefault(lbl, {"hit": 0, "sub": 0, "ins": 0, "del": 0})
        if e["op"] == "sub":
            time_bins[lbl]["sub"] += 1
        elif e["op"] == "ins":
            time_bins[lbl]["ins"] += 1
        elif e["op"] == "del":
            time_bins[lbl]["del"] += 1

    top = sorted(
        [{"window": k, **v} for k, v in time_bins.items()],
        key=lambda x: (x["sub"], x["ins"], x["del"]),
        reverse=True
    )[:10]

    return {
        "events": events,
        "time_bins": time_bins,
        "top_substitution_windows": top,
    }
