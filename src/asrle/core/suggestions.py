from __future__ import annotations

from asrle.types import AnalysisReport, Suggestion


def generate_suggestions(report: AnalysisReport) -> list[Suggestion]:
    out: list[Suggestion] = []

    # Latency suggestions
    if report.latency:
        total = report.latency.total_s
        stages = report.latency.stages_s
        if total > 4.0:
            out.append(
                Suggestion(
                    severity="warning",
                    title="High total latency",
                    rationale=f"Total transcription time is {total:.2f}s (wall-clock).",
                    action="Consider faster backend, quantization, smaller model, or reduce beam width / chunk length.",
                )
            )
        if stages:
            top = sorted(stages.items(), key=lambda kv: kv[1], reverse=True)[:2]
            for name, t in top:
                if t / max(1e-9, total) > 0.45:
                    out.append(
                        Suggestion(
                            severity="info",
                            title=f"Dominant stage: {name}",
                            rationale=f"Stage '{name}' accounts for {t:.2f}s ({(t/total)*100:.0f}%) of runtime.",
                            action=f"Optimize or change configuration affecting '{name}' (beam/search, device, batching).",
                        )
                    )

    # WER suggestions
    if report.wer:
        if report.wer.wer > 0.25:
            out.append(
                Suggestion(
                    severity="warning",
                    title="High WER",
                    rationale=f"WER is {report.wer.wer:.3f}.",
                    action="Check language/task settings, audio quality (SNR), and consider stronger model or domain adaptation.",
                )
            )

    # Drift suggestions
    if report.drift and report.drift.suspicious:
        out.append(
            Suggestion(
                severity="critical",
                title="Suspicious timestamp drift symptoms",
                rationale=f"Max segment timing irregularity is {report.drift.max_abs_drift_s:.2f}s.",
                action="Review chunk overlap/right-context; ensure monotonic timestamps; validate VAD segmentation.",
            )
        )

    # Attribution suggestions
    if report.attribution:
        worst = sorted(report.attribution, key=lambda s: s.wer, reverse=True)[:3]
        if worst and worst[0].wer > 0.5 and report.reference_text:
            out.append(
                Suggestion(
                    severity="info",
                    title="A few segments dominate the errors",
                    rationale="Top error segments show very high segment-level WER.",
                    action="Inspect these segments; add noise-robust preprocessing; consider VAD + resegmentation; check far-field/noise.",
                )
            )

    return out
