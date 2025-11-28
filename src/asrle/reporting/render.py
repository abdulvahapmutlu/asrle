from __future__ import annotations

from asrle.types import AnalysisReport


def render_markdown(report: AnalysisReport) -> str:
    lines: list[str] = []
    lines.append("# ASR-LE Report")
    lines.append("")
    lines.append(f"- **Backend:** `{report.backend}`")
    lines.append(f"- **Audio:** `{report.audio_path}`")
    if report.audio_duration_s is not None:
        lines.append(f"- **Duration:** {report.audio_duration_s:.2f}s")
    lines.append(f"- **Created:** {report.created_at_iso}")
    lines.append(f"- **Tool:** {report.tool_version}")
    lines.append("")

    if report.streaming:
        lines.append("## Streaming")
        lines.append("")
        lines.append(f"- **Wall total:** {report.streaming.wall_clock_total_s:.3f}s")
        if report.streaming.first_word_latency_s is not None:
            lines.append(f"- **First-word latency (sample):** {report.streaming.first_word_latency_s:.3f}s")
        if report.streaming.first_word_latency_percentiles_s:
            lines.append(
                "- **First-word latency percentiles:** "
                + ", ".join([f"{k}={v:.3f}s" for k, v in report.streaming.first_word_latency_percentiles_s.items()])
            )
        lines.append("")

    lines.append("## Transcript")
    lines.append("")
    lines.append(report.transcript.text if report.transcript.text else "_(empty)_")
    lines.append("")

    if report.wer:
        lines.append("## WER")
        lines.append("")
        lines.append(f"- **WER:** {report.wer.wer:.4f}")
        lines.append(f"- S/D/I: {report.wer.substitutions}/{report.wer.deletions}/{report.wer.insertions}")
        lines.append("")

    if report.latency:
        lines.append("## Latency")
        lines.append("")
        lines.append(f"- **Avg total:** {report.latency.total_s:.3f}s")
        if report.latency.percentiles_s:
            lines.append(
                "- **Percentiles:** "
                + ", ".join([f"{k}={v:.3f}s" for k, v in report.latency.percentiles_s.items()])
            )
        if report.latency.stages_s:
            lines.append("")
            lines.append("### Stages")
            for k, v in sorted(report.latency.stages_s.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"- `{k}`: {v:.3f}s")
        lines.append("")

    if report.moments and report.moments.moments:
        lines.append("## Top Error Moments (timestamp-only clips)")
        lines.append("")
        for m in report.moments.moments:
            lines.append(
                f"- **#{m.rank}** [{m.start_s:.2f}–{m.end_s:.2f}s] (bin `{m.window_label}`) "
                f"score={m.score:.1f} | sub={m.counts.get('sub',0)} ins={m.counts.get('ins',0)} del={m.counts.get('del',0)}"
            )
            if m.sample_events:
                lines.append("  - sample:")
                for e in m.sample_events[:5]:
                    lines.append(f"    - t={e.get('t'):.2f}s {e.get('op')} ref=`{e.get('ref')}` hyp=`{e.get('hyp')}`")
        lines.append("")

    if report.word_attribution:
        lines.append("## Word-level Error Attribution (timestamped)")
        lines.append("")
        lines.append("### Worst substitution windows")
        for w in report.word_attribution.top_substitution_windows[:10]:
            lines.append(f"- `{w['window']}` sub={w['sub']} ins={w['ins']} del={w['del']}")
        lines.append("")
        lines.append("### First 40 error events (non-hits)")
        shown = 0
        for e in report.word_attribution.events:
            if e.op == "hit":
                continue
            lines.append(
                f"- [{e.start_s:.2f}-{e.end_s:.2f}s] **{e.op.upper()}** "
                f"ref=`{e.ref_word}` hyp=`{e.hyp_word}`"
            )
            shown += 1
            if shown >= 40:
                break
        lines.append("")

    if report.drift:
        lines.append("## Timestamp Drift Heuristics")
        lines.append("")
        lines.append(f"- **Suspicious:** {report.drift.suspicious}")
        lines.append(f"- **Max abs drift symptom:** {report.drift.max_abs_drift_s:.3f}s")
        if report.drift.reasons:
            lines.append("")
            lines.append("### Reasons")
            for r in report.drift.reasons[:12]:
                lines.append(f"- {r}")
        lines.append("")

    if report.attribution:
        lines.append("## Error Attribution (Segment Level)")
        lines.append("")
        worst = sorted(report.attribution, key=lambda s: s.wer, reverse=True)[:10]
        for s in worst:
            lines.append(f"### [{s.start_s:.2f}s → {s.end_s:.2f}s] WER={s.wer:.3f}")
            if s.ref is not None:
                lines.append(f"- **REF:** {s.ref}")
            lines.append(f"- **HYP:** {s.hyp}")
            if s.notes:
                lines.append(f"- Notes: {', '.join(s.notes)}")
            lines.append("")

    if report.suggestions:
        lines.append("## Suggestions")
        lines.append("")
        for sug in report.suggestions:
            lines.append(f"### ({sug.severity}) {sug.title}")
            lines.append(f"- Rationale: {sug.rationale}")
            lines.append(f"- Action: {sug.action}")
            lines.append("")

    if report.traces:
        lines.append("## Traces")
        lines.append("")
        for k, v in report.traces.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    return "\n".join(lines)


def write_report_markdown(report: AnalysisReport, out_dir: str) -> str:
    import os

    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return path
