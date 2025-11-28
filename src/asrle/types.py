from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class WordStamp:
    word: str
    start_s: float
    end_s: float
    confidence: float | None = None
    source: str | None = None  # e.g. "backend", "ctc-aligner", "heuristic"


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    text: str
    tokens: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[Segment]
    language: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # may include "words": list[WordStamp-like dict]


@dataclass(frozen=True)
class LatencyBreakdown:
    total_s: float
    stages_s: dict[str, float] = field(default_factory=dict)
    percentiles_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    hyp_words: int


@dataclass(frozen=True)
class AttributedSpan:
    start_s: float
    end_s: float
    hyp: str
    ref: str | None
    wer: float
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DriftPoint:
    t_s: float
    drift_s: float
    note: str | None = None


@dataclass(frozen=True)
class DriftReport:
    drift_points: list[DriftPoint]
    max_abs_drift_s: float
    suspicious: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Suggestion:
    severity: Literal["info", "warning", "critical"]
    title: str
    rationale: str
    action: str


@dataclass(frozen=True)
class StreamingConfig:
    enabled: bool = False
    chunk_ms: int = 800
    overlap_ms: int = 160
    right_context_ms: int = 0  # lookahead audio included, but trimmed from outputs
    max_chunks: int | None = None


@dataclass(frozen=True)
class StreamingReport:
    config: StreamingConfig
    transcript: Transcript

    wall_clock_total_s: float
    per_chunk_wall_s: list[float] = field(default_factory=list)

    # first token emission latency (wall-clock)
    first_word_latency_s: float | None = None
    first_word_latency_percentiles_s: dict[str, float] = field(default_factory=dict)

    merge_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WordErrorEvent:
    op: Literal["hit", "sub", "ins", "del"]
    start_s: float
    end_s: float
    ref_word: str | None
    hyp_word: str | None
    ref_start_s: float | None = None
    ref_end_s: float | None = None
    hyp_start_s: float | None = None
    hyp_end_s: float | None = None
    confidence: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class WordAttributionReport:
    events: list[WordErrorEvent] = field(default_factory=list)
    # { "0-1s": {"sub":2,"ins":1,...} }
    time_bins: dict[str, dict[str, int]] = field(default_factory=dict)
    top_substitution_windows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ErrorMoment:
    """
    Timestamp-only "clip suggestion": where errors spike.
    """
    rank: int
    start_s: float
    end_s: float
    window_label: str
    counts: dict[str, int]  # sub/ins/del/hit
    score: float
    sample_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MomentsReport:
    bin_s: float
    pad_s: float
    moments: list[ErrorMoment] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisReport:
    backend: str
    backend_params: dict[str, Any]
    audio_path: str
    audio_duration_s: float | None
    transcript: Transcript
    reference_text: str | None = None

    latency: LatencyBreakdown | None = None
    wer: WERResult | None = None
    attribution: list[AttributedSpan] = field(default_factory=list)
    drift: DriftReport | None = None
    suggestions: list[Suggestion] = field(default_factory=list)

    # word-level: keep both sides
    hyp_words: list[WordStamp] = field(default_factory=list)
    ref_words: list[WordStamp] = field(default_factory=list)
    word_attribution: WordAttributionReport | None = None

    # NEW: top error moments (timestamp-only "clips")
    moments: MomentsReport | None = None

    # streaming
    streaming: StreamingReport | None = None

    # profiler artifacts
    traces: dict[str, str] = field(default_factory=dict)

    created_at_iso: str = ""
    tool_version: str = ""


@dataclass(frozen=True)
class DatasetItemResult:
    audio_path: str
    ok: bool
    backend: str
    duration_s: float | None
    wer: float | None
    p95_latency_s: float | None
    first_word_latency_s: float | None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    out_dir: str | None = None


@dataclass(frozen=True)
class DatasetSummary:
    backend: str
    backend_params: dict[str, Any]
    n: int
    ok: int
    failed: int

    wer_mean: float | None
    wer_p50: float | None
    wer_p90: float | None

    latency_p50_s: float | None
    latency_p95_s: float | None

    fwlat_p50_s: float | None
    fwlat_p95_s: float | None

    slices: dict[str, Any] = field(default_factory=dict)
    items: list[DatasetItemResult] = field(default_factory=list)
