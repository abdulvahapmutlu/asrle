from __future__ import annotations

from asrle.types import WERResult
from asrle.utils.text import normalize_text, split_words


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    """
    Uses jiwer if available (best), else a simple Levenshtein word-distance fallback.

    Fix: normalize both inputs (incl. BOM removal) before jiwer.
    """
    reference_n = normalize_text(reference)
    hypothesis_n = normalize_text(hypothesis)

    try:
        from jiwer import process_words  # type: ignore

        out = process_words(reference_n, hypothesis_n)

        # jiwer field names differ by version; read robustly.
        ref_wc = getattr(out, "reference_word_count", getattr(out, "references", 0))
        hyp_wc = getattr(out, "hypothesis_word_count", getattr(out, "hypotheses", 0))

        return WERResult(
            wer=float(out.wer),
            substitutions=int(out.substitutions),
            deletions=int(out.deletions),
            insertions=int(out.insertions),
            hits=int(out.hits),
            ref_words=int(ref_wc),
            hyp_words=int(hyp_wc),
        )
    except Exception:
        ref = split_words(reference_n)
        hyp = split_words(hypothesis_n)

        # DP Levenshtein distance at word level
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,          # deletion
                    dp[i][j - 1] + 1,          # insertion
                    dp[i - 1][j - 1] + cost,   # substitution/hit
                )
        dist = dp[n][m]
        wer = dist / max(1, n)

        # Fallback counts are unknown precisely without backtrace; provide coarse counts.
        return WERResult(
            wer=float(wer),
            substitutions=int(dist),
            deletions=0,
            insertions=0,
            hits=max(0, n - dist),
            ref_words=n,
            hyp_words=m,
        )
