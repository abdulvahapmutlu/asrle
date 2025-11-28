from __future__ import annotations

import os
from typing import Any

import streamlit as st

from asrle.dashboard.webapp_utils import read_json, slugify, zip_dir


def _metric(v: Any, fmt: str) -> str:
    if v is None:
        return "n/a"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def _best_duration_s(r: dict[str, Any], timeline_bins: list[dict[str, Any]] | None) -> float:
    dur = r.get("audio_duration_s")
    if isinstance(dur, (int, float)) and dur > 0:
        return float(dur)

    if timeline_bins:
        try:
            return float(max(float(b.get("end_s", 0.0)) for b in timeline_bins))
        except Exception:
            pass

    # fallback
    return 1.0


def _render_transcript_with_timeline_overlay(
    *,
    transcript_text: str,
    timeline_bins: list[dict[str, Any]] | None,
    moments: list[dict[str, Any]] | None,
    duration_s: float,
) -> None:
    """
    Shows transcript, then a mini timeline bar directly under it.
    Timeline encodes substitution density (sub/ins/del),
    and shows moment markers aligned by time.
    """
    st.subheader("Transcript")
    st.code(transcript_text or "", language="text")

    if not timeline_bins:
        return

    # compute intensity from substitutions (dominant signal)
    subs = []
    for b in timeline_bins:
        try:
            subs.append(float(b.get("sub", 0.0)))
        except Exception:
            subs.append(0.0)
    max_sub = max(subs) if subs else 0.0

    # build HTML overlay
    st.markdown(
        """
<style>
.asrle-wrap { margin-top: 8px; }
.asrle-tlbox { position: relative; width: 100%; }
.asrle-tl {
  display: flex;
  height: 14px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(0,0,0,0.03);
}
.asrle-bin { flex: 1; min-width: 2px; }
.asrle-scale {
  display:flex;
  justify-content: space-between;
  font-size: 12px;
  opacity: 0.75;
  margin-top: 4px;
}
.asrle-marker {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 2px;
  border-radius: 2px;
  background: rgba(0,0,0,0.35);
}
.asrle-marker::after{
  content: "";
  position: absolute;
  top: -6px;
  left: -4px;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.35);
}
.asrle-marker-label{
  position: absolute;
  top: -26px;
  left: 6px;
  font-size: 12px;
  opacity: 0.85;
  white-space: nowrap;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # bins row (background intensity = substitution density)
    bins_html = []
    for i, b in enumerate(timeline_bins):
        label = str(b.get("label", f"bin{i}"))
        sub = int(b.get("sub", 0))
        ins = int(b.get("ins", 0))
        dele = int(b.get("del", 0))

        # intensity: neutral gray -> red-ish as subs grow
        if max_sub <= 0:
            alpha = 0.0
        else:
            alpha = min(1.0, float(sub) / float(max_sub))

        # keep it readable even for small counts
        base = 0.06
        amp = 0.80
        a = base + amp * alpha

        style = f"background: rgba(220, 20, 60, {a:.3f});"
        title = f"{label} | sub={sub} ins={ins} del={dele}"
        bins_html.append(f"<div class='asrle-bin' title='{title}' style='{style}'></div>")

    # moment markers (vertical lines)
    markers_html = []
    if moments:
        for m in moments:
            try:
                s0 = float(m.get("start_s", 0.0))
                s1 = float(m.get("end_s", 0.0))
                rank = int(m.get("rank", 0))
                center = 0.5 * (s0 + s1)
                pct = 0.0 if duration_s <= 0 else max(0.0, min(1.0, center / duration_s))
                left = pct * 100.0
                lab = f"#{rank}"
                markers_html.append(
                    f"<div class='asrle-marker' style='left:{left:.2f}%;'>"
                    f"<div class='asrle-marker-label'>{lab}</div>"
                    f"</div>"
                )
            except Exception:
                continue

    html = (
        "<div class='asrle-wrap'>"
        "<div class='asrle-tlbox'>"
        f"<div class='asrle-tl'>{''.join(bins_html)}</div>"
        f"{''.join(markers_html)}"
        "</div>"
        f"<div class='asrle-scale'><div>0.0s</div><div>{duration_s:.2f}s</div></div>"
        "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)


def render_run_report(run_dir: str) -> None:
    report_path = os.path.join(run_dir, "report.json")
    if not os.path.exists(report_path):
        st.warning("report.json not found in this directory.")
        return

    r = read_json(report_path)

    run_uid = slugify(os.path.basename(os.path.abspath(run_dir)))
    event_range_key = f"event_range__{run_uid}"

    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Backend", str(r.get("backend")))
    with c2:
        dur = r.get("audio_duration_s")
        st.metric("Duration", _metric(dur, "{:.2f}s"))
    with c3:
        wer = (r.get("wer") or {}).get("wer")
        st.metric("WER", _metric(wer, "{:.4f}"))
    with c4:
        p95 = ((r.get("latency") or {}).get("percentiles_s") or {}).get("p95")
        st.metric("Latency p95", _metric(p95, "{:.3f}s"))

    # Load artifacts (timeline + moments)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    timeline_bins: list[dict[str, Any]] | None = None
    moments: list[dict[str, Any]] | None = None

    timeline_path = os.path.join(artifacts_dir, "timeline.json")
    if os.path.exists(timeline_path):
        try:
            timeline_bins = (read_json(timeline_path).get("bins") or [])[:]
        except Exception:
            timeline_bins = None

    moments_path = os.path.join(artifacts_dir, "error_moments.json")
    if os.path.exists(moments_path):
        try:
            moments = (read_json(moments_path).get("moments") or [])[:]
        except Exception:
            moments = None

    duration_s = _best_duration_s(r, timeline_bins)

    # Transcript + overlay timeline directly under text
    transcript_text = (r.get("transcript") or {}).get("text", "")
    _render_transcript_with_timeline_overlay(
        transcript_text=transcript_text,
        timeline_bins=timeline_bins,
        moments=moments,
        duration_s=duration_s,
    )

    # Streaming summary
    if r.get("streaming"):
        st.subheader("Streaming")
        sr = r["streaming"]
        fw_p95 = (sr.get("first_word_latency_percentiles_s") or {}).get("p95")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wall total", _metric(sr.get("wall_clock_total_s"), "{:.3f}s"))
        with c2:
            st.metric("First word (sample)", _metric(sr.get("first_word_latency_s"), "{:.3f}s"))
        with c3:
            st.metric("First word p95", _metric(fw_p95, "{:.3f}s"))
        with st.expander("Streaming details"):
            st.json(sr)

    # Heatmap (keep existing view, still useful)
    if timeline_bins:
        st.subheader("Error Timeline Heatmap (sub/ins/del over time)")
        _render_heatmap(timeline_bins)

    # ---- Clickable moments: set time slider + filter table automatically ----
    if moments:
        st.subheader("Top Error Moments (click to zoom/filter)")
        cols = st.columns(min(3, max(1, len(moments))))
        for i, m in enumerate(moments[:6]):  # show first 6; top-3 is default but allow more
            c = cols[i % len(cols)]
            with c:
                try:
                    rank = int(m.get("rank", i + 1))
                    s0 = float(m.get("start_s", 0.0))
                    s1 = float(m.get("end_s", 0.0))
                    counts = m.get("counts", {}) or {}
                    sub = int(counts.get("sub", 0))
                    ins = int(counts.get("ins", 0))
                    dele = int(counts.get("del", 0))
                    score = float(m.get("score", 0.0))

                    label = f"🎯 #{rank}  {s0:.2f}–{s1:.2f}s"
                    small = f"score={score:.1f} | sub={sub} ins={ins} del={dele}"

                    if st.button(label, key=f"moment_btn__{run_uid}__{rank}", use_container_width=True):
                        st.session_state[event_range_key] = (s0, s1)
                        st.rerun()

                    st.caption(small)
                except Exception:
                    st.json(m)

        with st.expander("Moments JSON (debug)"):
            st.json(moments)

    # ---- Word attribution + auto-filtered table ----
    if r.get("word_attribution"):
        st.subheader("Word-level Attribution (timestamped)")

        wa = r["word_attribution"]
        events = wa.get("events", []) or []
        non_hits = [e for e in events if e.get("op") != "hit"]

        st.write("Worst substitution windows:")
        st.json((wa.get("top_substitution_windows") or [])[:10])

        # compute max_t from events/end or duration
        max_t = float(duration_s)
        for e in events:
            try:
                max_t = max(max_t, float(e.get("end_s", 0.0)))
            except Exception:
                pass
        max_t = float(max(1.0, max_t))

        # default range
        if event_range_key not in st.session_state:
            st.session_state[event_range_key] = (0.0, max_t)

        # If duration changed, clamp
        cur0, cur1 = st.session_state[event_range_key]
        cur0 = max(0.0, float(cur0))
        cur1 = min(max_t, float(cur1))
        if cur1 < cur0:
            cur0, cur1 = 0.0, max_t
        st.session_state[event_range_key] = (cur0, cur1)

        # The slider is the *single source of truth*. Moments update it via session_state + rerun.
        t0, t1 = st.slider(
            "Filter events by time (seconds)",
            min_value=0.0,
            max_value=max_t,
            value=st.session_state[event_range_key],
            key=event_range_key,
        )

        # Filter events (auto-updates)
        filt = []
        for e in non_hits:
            try:
                ts = float(e.get("start_s", 0.0))
                if t0 <= ts <= t1:
                    filt.append(e)
            except Exception:
                continue

        # Helpful summary
        st.caption(f"Showing {min(len(filt), 400)} non-hit events in [{t0:.2f}, {t1:.2f}] seconds")

        # Show table
        st.dataframe(filt[:400], use_container_width=True)

        # Optional: show all events in window (including hits) as density
        with st.expander("Window stats"):
            sub = sum(1 for e in filt if e.get("op") == "sub")
            ins = sum(1 for e in filt if e.get("op") == "ins")
            dele = sum(1 for e in filt if e.get("op") == "del")
            st.write({"sub": sub, "ins": ins, "del": dele, "total": len(filt)})

    # Segment-level attribution
    if r.get("attribution"):
        st.subheader("Segment-level Error Attribution")
        spans = r.get("attribution") or []
        spans = sorted(spans, key=lambda x: float(x.get("wer", 0.0)), reverse=True)
        st.dataframe(spans[:30], use_container_width=True)

    # Drift + suggestions
    if r.get("drift"):
        st.subheader("Timestamp Drift Heuristics")
        st.json(r["drift"])

    if r.get("suggestions"):
        st.subheader("Suggestions")
        for sug in r["suggestions"]:
            st.markdown(f"### ({sug['severity']}) {sug['title']}")
            st.write("Rationale:", sug.get("rationale"))
            st.write("Action:", sug.get("action"))

    # Traces
    if r.get("traces"):
        st.subheader("Traces")
        st.json(r["traces"])

    # Downloads
    st.subheader("Downloads")
    _render_downloads(run_dir)


def render_dataset_report(run_dir: str) -> None:
    path = os.path.join(run_dir, "dataset_summary.json")
    if not os.path.exists(path):
        st.warning("dataset_summary.json not found in this directory.")
        return

    s = read_json(path)
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Items", str(s.get("n")))
    with c2:
        st.metric("OK", str(s.get("ok")))
    with c3:
        st.metric("Failed", str(s.get("failed")))
    with c4:
        st.metric("WER mean", _metric(s.get("wer_mean"), "{:.4f}"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("WER p50", _metric(s.get("wer_p50"), "{:.4f}"))
    with c2:
        st.metric("WER p90", _metric(s.get("wer_p90"), "{:.4f}"))
    with c3:
        st.metric("Latency p95", _metric(s.get("latency_p95_s"), "{:.3f}s"))
    with c4:
        st.metric("FW Lat p95", _metric(s.get("fwlat_p95_s"), "{:.3f}s"))

    st.subheader("Items")
    items = s.get("items") or []
    show_failed = st.checkbox("Show failed only", value=False)
    if show_failed:
        items = [it for it in items if not it.get("ok")]
    st.dataframe(items[:500], use_container_width=True)

    st.subheader("Downloads")
    _render_downloads(run_dir)


def _render_downloads(run_dir: str) -> None:
    rp = os.path.join(run_dir, "report.json")
    if os.path.exists(rp):
        with open(rp, "rb") as f:
            st.download_button("Download report.json", data=f, file_name="report.json", mime="application/json")

    md = os.path.join(run_dir, "report.md")
    if os.path.exists(md):
        with open(md, "rb") as f:
            st.download_button("Download report.md", data=f, file_name="report.md", mime="text/markdown")

    if st.button("Prepare ZIP of run directory"):
        z = zip_dir(run_dir)
        with open(z, "rb") as f:
            st.download_button(
                "Download run.zip",
                data=f,
                file_name=os.path.basename(run_dir) + ".zip",
                mime="application/zip",
            )


def _render_heatmap(timeline_bins: list[dict[str, Any]]) -> None:
    if not timeline_bins:
        st.info("No bins available.")
        return

    ops = ["sub", "ins", "del"]
    mat = []
    labels = [b.get("label", "") for b in timeline_bins]

    for op in ops:
        mat.append([float(b.get(op, 0)) for b in timeline_bins])

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(ops)))
        ax.set_yticklabels(ops)
        ax.set_title("Word error density over time")
        ax.set_xlabel("time bins")

        n = len(labels)
        stride = max(1, n // 12)
        xs = list(range(0, n, stride))
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[i] for i in xs], rotation=45, ha="right")

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.json({"bins": timeline_bins[:30]})
