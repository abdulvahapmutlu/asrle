from __future__ import annotations

import datetime as dt
import os
import re

from asrle.backends.registry import BackendRegistry
from asrle.core.alignment.ctc import CTCAlignerConfig, CTCWordAligner
from asrle.core.alignment.reconcile import reconcile_words
from asrle.core.drift import detect_timestamp_drift
from asrle.core.error_attribution import attribute_errors_by_segment
from asrle.core.metrics import compute_wer
from asrle.core.moments import build_error_moments
from asrle.core.profiler import Profiler
from asrle.core.streaming.simulator import StreamingSimulator
from asrle.core.suggestions import generate_suggestions
from asrle.core.word_attribution import build_word_attribution, get_hyp_words
from asrle.reporting.artifacts import write_artifacts
from asrle.reporting.report import write_report_json
from asrle.reporting.render import write_report_markdown
from asrle.reporting.timeline import build_timeline_series
from asrle.types import (
    AnalysisReport,
    LatencyBreakdown,
    MomentsReport,
    StreamingConfig,
    Transcript,
    WordAttributionReport,
    WordStamp,
)
from asrle.utils.audio import probe_audio
from asrle.utils.io import ensure_dir, write_json
from asrle.utils.text import normalize_text  # <-- added (minimal)
from asrle.utils.time import Timer
from asrle.version import __version__


def _extract_ref_words_ctc(
    audio_path: str,
    reference_text: str,
    model_name: str,
    device: str,
) -> list[WordStamp]:
    aligner = CTCWordAligner(CTCAlignerConfig(model_name=model_name, device=device))
    aligned = aligner.align_words(audio_path, reference_text)
    return reconcile_words(aligned)


def _percentiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    v = sorted(float(x) for x in vals)
    if len(v) == 1:
        return {"p50": v[0], "p95": v[0], "p99": v[0]}

    def pick(q: float) -> float:
        idx = int(round(q * (len(v) - 1)))
        idx = max(0, min(len(v) - 1, idx))
        return float(v[idx])

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def _ctc_alignment_looks_bad(ref_words: list[WordStamp], audio_duration_s: float | None) -> bool:
    """
    Minimal safety gate:
    If CTC tokenization collapses into '<unk>' / empty tokens, word attribution becomes garbage.
    In that case we skip word attribution rather than producing fake substitution windows.
    """
    if not ref_words:
        return True

    norm_words = [normalize_text(w.word) for w in ref_words]
    if not norm_words:
        return True

    unk = sum(1 for w in norm_words if (not w) or w in {"<unk>", "unk"})
    if (unk / max(1, len(norm_words))) >= 0.20:
        return True

    if audio_duration_s:
        try:
            max_end = max(float(w.end_s) for w in ref_words)
            if max_end < float(audio_duration_s) * 0.70:
                return True
        except Exception:
            return True

    return False


class Engine:
    def __init__(self, registry: BackendRegistry | None = None):
        self.registry = registry or BackendRegistry()

    def analyze(
        self,
        audio_path: str,
        backend_name: str,
        backend_params: dict,
        reference_text: str | None = None,
        repeats: int = 1,
        enable_profiling: bool = True,
        compute_wer_flag: bool = True,
        compute_attribution_flag: bool = True,
        compute_drift_flag: bool = True,
        enable_suggestions_flag: bool = True,
        # word alignment + word attribution
        word_align: bool = False,
        aligner_model: str = "facebook/wav2vec2-base-960h",
        aligner_device: str = "cpu",
        word_attribution: bool = True,
        # streaming simulation
        streaming_enabled: bool = False,
        chunk_ms: int = 800,
        overlap_ms: int = 160,
        right_context_ms: int = 0,
        # outputs
        out_dir: str | None = None,
        # torch trace (best-effort)
        torch_trace: bool = False,
        torch_trace_dirname: str = "torch_trace",
    ) -> AnalysisReport:
        params = dict(backend_params or {})
        backend = None
        for _ in range(8):
            try:
                backend = self.registry.create(backend_name, **params)
                break
            except TypeError as e:
                msg = str(e)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad = m.group(1)
                if bad in params:
                    del params[bad]
                    continue
                raise

        assert backend is not None
        backend_params = params
        audio_info = probe_audio(audio_path)

        trace_paths: dict[str, str] = {}

        streaming_report = None
        transcript: Transcript | None = None
        total_times: list[float] = []
        fwlat_samples: list[float] = []

        # ---- Streaming mode (repeat to estimate p95 first-word latency) ----
        if streaming_enabled:
            sim = StreamingSimulator(
                StreamingConfig(enabled=True, chunk_ms=chunk_ms, overlap_ms=overlap_ms, right_context_ms=right_context_ms)
            )
            for _ in range(max(1, repeats)):
                with Timer("streaming_total") as tt:
                    sr = sim.simulate(backend, audio_path)
                total_times.append(tt.elapsed_s)
                streaming_report = sr
                transcript = sr.transcript
                if sr.first_word_latency_s is not None:
                    fwlat_samples.append(float(sr.first_word_latency_s))

            assert streaming_report is not None and transcript is not None
            streaming_report = type(streaming_report)(
                **{
                    **streaming_report.__dict__,
                    "first_word_latency_percentiles_s": _percentiles(fwlat_samples),
                }
            )
            latency = LatencyBreakdown(
                total_s=float(sum(total_times) / len(total_times)),
                stages_s={},
                percentiles_s=_percentiles(total_times),
            )

        # ---- Non-streaming mode ----
        else:
            prof_total = Profiler()
            local_times: list[float] = []
            for _ in range(max(1, repeats)):
                prof = Profiler()

                def run() -> Transcript:
                    if not torch_trace:
                        return backend.transcribe(audio_path, profiler=prof if enable_profiling else None)

                    try:
                        import torch

                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if torch.cuda.is_available():
                            activities.append(torch.profiler.ProfilerActivity.CUDA)

                        out = None
                        with torch.profiler.profile(
                            activities=activities,
                            record_shapes=False,
                            with_stack=True,
                        ) as p:
                            out = backend.transcribe(audio_path, profiler=prof if enable_profiling else None)

                        if out_dir:
                            td = os.path.join(out_dir, torch_trace_dirname)
                            ensure_dir(td)
                            trace_file = os.path.join(td, "trace.json")
                            p.export_chrome_trace(trace_file)
                            trace_paths["torch_trace"] = trace_file

                        return out
                    except Exception:
                        return backend.transcribe(audio_path, profiler=prof if enable_profiling else None)

                with Timer("total") as t:
                    t_out = run()
                local_times.append(t.elapsed_s)
                transcript = t_out

                if enable_profiling:
                    for k, v in prof.stages_s.items():
                        prof_total.stages_s[k] = prof_total.stages_s.get(k, 0.0) + v

            assert transcript is not None
            latency = (
                LatencyBreakdown(
                    total_s=float(sum(local_times) / len(local_times)),
                    stages_s={k: float(v / max(1, repeats)) for k, v in prof_total.stages_s.items()},
                    percentiles_s=_percentiles(local_times),
                )
                if enable_profiling and local_times
                else None
            )

        # ---- Metrics ----
        wer_res = compute_wer(reference_text, transcript.text) if (compute_wer_flag and reference_text) else None
        attribution = attribute_errors_by_segment(transcript.segments, reference_text) if compute_attribution_flag else []
        drift = detect_timestamp_drift(transcript.segments) if compute_drift_flag else None

        # ---- Word stamps (hyp) ----
        hyp_words = reconcile_words(get_hyp_words(transcript))

        # ---- Word stamps (ref) ----
        ref_words: list[WordStamp] = []
        if reference_text and (word_align or word_attribution):
            try:
                ref_words = _extract_ref_words_ctc(
                    audio_path=audio_path,
                    reference_text=reference_text,
                    model_name=aligner_model,
                    device=aligner_device,
                )
                # MINIMAL FIX: if alignment is garbage (<unk>-heavy / too-short), drop it.
                if _ctc_alignment_looks_bad(ref_words, audio_info.duration_s):
                    ref_words = []
            except Exception:
                ref_words = []

        # ---- Word-level attribution (timestamped) ----
        word_attr: WordAttributionReport | None = None
        moments: MomentsReport | None = None
        if word_attribution and reference_text and ref_words and hyp_words:
            try:
                word_attr = build_word_attribution(ref_words=ref_words, hyp_words=hyp_words, bin_s=1.0)
                moments = build_error_moments(word_attr, top_k=3, bin_s=1.0, pad_s=0.6)
            except Exception:
                word_attr = None
                moments = None

        report = AnalysisReport(
            backend=backend_name,
            backend_params=dict(backend_params),
            audio_path=audio_path,
            audio_duration_s=audio_info.duration_s,
            transcript=transcript,
            reference_text=reference_text,
            latency=latency,
            wer=wer_res,
            attribution=attribution,
            drift=drift,
            suggestions=[],
            hyp_words=hyp_words,
            ref_words=ref_words,
            word_attribution=word_attr,
            moments=moments,
            streaming=streaming_report,
            traces=trace_paths,
            created_at_iso=dt.datetime.now(dt.timezone.utc).isoformat(),
            tool_version=__version__,
        )

        if enable_suggestions_flag:
            report = AnalysisReport(**{**report.__dict__, "suggestions": generate_suggestions(report)})

        # ---- Persist outputs ----
        if out_dir:
            ensure_dir(out_dir)
            write_report_json(report, out_dir)
            write_report_markdown(report, out_dir)
            write_artifacts(report, out_dir)

            art_dir = os.path.join(out_dir, "artifacts")
            ensure_dir(art_dir)

            if report.hyp_words:
                write_json(os.path.join(art_dir, "hyp_words.json"), {"words": [w.__dict__ for w in report.hyp_words]})
            if report.ref_words:
                write_json(os.path.join(art_dir, "ref_words.json"), {"words": [w.__dict__ for w in report.ref_words]})
            if report.word_attribution:
                write_json(os.path.join(art_dir, "word_attribution.json"), report.word_attribution)
                write_json(os.path.join(art_dir, "timeline.json"), {"bins": build_timeline_series(report.word_attribution)})
            if report.moments:
                write_json(os.path.join(art_dir, "error_moments.json"), report.moments)

        return report
