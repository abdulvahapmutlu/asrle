from __future__ import annotations

from asrle.core.metrics import compute_wer
from asrle.types import AttributedSpan, Segment
from asrle.utils.text import normalize_text


def attribute_errors_by_segment(
    segments: list[Segment],
    reference_text: str | None,
) -> list[AttributedSpan]:
    """
    Segment-level attribution:
    - If reference is provided, it splits reference into a rough same-count partition.
      (Advanced option: replace with forced alignment / ASR aligner later.)
    - Computes WER per segment and ranks segments by error.
    """
    if not reference_text:
        return [
            AttributedSpan(
                start_s=s.start_s,
                end_s=s.end_s,
                hyp=s.text,
                ref=None,
                wer=0.0,
                notes=["No reference provided; attribution limited."],
            )
            for s in segments
        ]

    ref_norm = normalize_text(reference_text)
    hyp_norm = normalize_text(" ".join([s.text for s in segments]))

    # Basic safeguard: if hypothesis is empty, assign worst.
    if not hyp_norm:
        return [
            AttributedSpan(s.start_s, s.end_s, s.text, reference_text, 1.0, ["Empty hypothesis"])
            for s in segments
        ]

    # Rough partition strategy: map reference words to segments proportional to hyp words per segment.
    ref_words = ref_norm.split()
    hyp_words_per_seg = [len(normalize_text(s.text).split()) for s in segments]
    total_hyp_words = max(1, sum(hyp_words_per_seg))

    spans: list[AttributedSpan] = []
    cursor = 0
    for seg, hw in zip(segments, hyp_words_per_seg):
        take = int(round((hw / total_hyp_words) * len(ref_words)))
        take = max(1, take) if len(ref_words) > 0 else 0
        ref_slice = " ".join(ref_words[cursor : min(len(ref_words), cursor + take)])
        cursor += take

        wer = compute_wer(ref_slice, seg.text).wer if ref_slice else 1.0
        notes: list[str] = []
        if seg.end_s - seg.start_s > 15:
            notes.append("Long segment; consider chunking/segmentation for more stable decoding.")
        spans.append(
            AttributedSpan(
                start_s=seg.start_s,
                end_s=seg.end_s,
                hyp=seg.text,
                ref=ref_slice if ref_slice else None,
                wer=float(wer),
                notes=notes,
            )
        )

    # Sort by worst segments first in the report consumer, but keep original order here.
    return spans
