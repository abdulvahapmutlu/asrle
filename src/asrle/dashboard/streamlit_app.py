from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from asrle.dashboard.webapp_actions import (
    get_engine_and_registry,
    run_dataset_web,
    run_single_analysis,
    validate_backend_web,
)
from asrle.dashboard.webapp_utils import (
    ensure_dir,
    human_relpath,
    list_run_dirs,
    now_run_id,
    read_json,
    save_uploaded_file,
    slugify,
)
from asrle.dashboard.webapp_views import render_dataset_report, render_run_report


def _backend_param_editor(backend_name: str) -> dict[str, Any]:
    """
    UI for backend params. For known backends, surface common knobs.
    Also accepts a JSON override.
    """
    st.caption("Backend parameters")
    params: dict[str, Any] = {}

    device_choice = st.selectbox("device", ["(auto/none)", "cpu", "cuda"], index=0)
    if device_choice != "(auto/none)":
        params["device"] = device_choice

    lang = st.text_input("language (optional)", value="").strip()
    if lang:
        params["language"] = lang

    if backend_name in {"hf-whisper", "faster-whisper"}:
        with st.expander("Backend-specific params", expanded=True):
            if backend_name == "hf-whisper":
                params["model_name"] = st.text_input("model_name", value="openai/whisper-small")
                params["task"] = st.selectbox("task", ["transcribe", "translate"], index=0)
                ch = st.text_input("chunk_length_s (optional)", value="").strip()
                params["chunk_length_s"] = float(ch) if ch else None
            else:
                params["model_name"] = st.text_input("model_name", value="small")
                params["compute_type"] = st.selectbox(
                    "compute_type", ["int8", "int8_float16", "float16", "float32"], index=0
                )
                params["beam_size"] = int(st.number_input("beam_size", min_value=1, max_value=50, value=5, step=1))
                params["vad_filter"] = bool(st.checkbox("vad_filter", value=False))

    elif backend_name == "dummy":
        with st.expander("Backend-specific params", expanded=True):
            params["text"] = st.text_input("dummy text", value="hello world")
            params["seconds"] = float(
                st.number_input("dummy duration seconds", min_value=0.1, max_value=60.0, value=2.0, step=0.1)
            )

    with st.expander("Advanced: JSON params override", expanded=False):
        raw = st.text_area("Paste JSON to merge/override params", value="{}", height=120)
        try:
            j = json.loads(raw or "{}")
            if isinstance(j, dict):
                params.update(j)
            else:
                st.warning("JSON override must be an object/dict.")
        except Exception as e:
            st.warning(f"Invalid JSON: {e}")

    params = {k: v for k, v in params.items() if v is not None}
    return params


def _reference_editor() -> str | None:
    st.caption("Reference transcript (optional, enables WER + alignment + moments)")
    mode = st.radio("Reference input", ["None", "Paste text", "Upload .txt"], horizontal=True, index=0)
    if mode == "None":
        return None
    if mode == "Paste text":
        ref = st.text_area("Reference text", value="", height=160).strip()
        return ref or None

    up = st.file_uploader("Upload reference .txt", type=["txt"])
    if not up:
        return None
    txt = up.getvalue().decode("utf-8", errors="ignore").strip()
    return txt or None


def _audio_input(base_runs_dir: str) -> str | None:
    st.caption("Audio input")
    mode = st.radio("Audio source", ["Upload", "Server path"], horizontal=True, index=0)

    if mode == "Upload":
        up = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a", "ogg"])
        if not up:
            return None
        td = ensure_dir(os.path.join(base_runs_dir, "_uploads"))
        out = os.path.join(td, slugify(up.name))
        save_uploaded_file(up, out)
        return out

    p = st.text_input("Audio path on server", value="").strip()
    return p or None


def _sidebar_global(default_base_runs_dir: str) -> dict[str, Any]:
    st.sidebar.header("ASR-LE Web App")
    st.sidebar.caption("Advanced ASR evaluation: timestamps, drift, moments, streaming latency, batch runs.")

    st.sidebar.subheader("Storage")
    base = st.sidebar.text_input("Base runs directory", value=default_base_runs_dir).strip() or default_base_runs_dir
    ensure_dir(base)
    return {"base_runs_dir": base}


def _f(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _parse_window_to_start_end(window: Any) -> tuple[float | None, float | None]:
    # Supports: "1-2s", "1-2", {"start_s":..,"end_s":..}, {"t0_s":..,"t1_s":..}
    if isinstance(window, dict):
        s = _f(window.get("start_s", window.get("t0_s")))
        e = _f(window.get("end_s", window.get("t1_s")))
        return s, e

    if isinstance(window, str):
        w = window.strip().lower().replace("sec", "s").replace("seconds", "s")
        w = w.replace(" ", "")
        if w.endswith("s"):
            w = w[:-1]
        # now "1-2" maybe
        if "-" in w:
            a, b = w.split("-", 1)
            return _f(a), _f(b)
    return None, None


def _extract_word_tokens(report: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = report.get("transcript") if isinstance(report, dict) else None
    if not isinstance(transcript, dict):
        return []

    rows: list[dict[str, Any]] = []
    segments = transcript.get("segments") if isinstance(transcript.get("segments"), list) else []
    for si, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        toks = seg.get("tokens") if isinstance(seg.get("tokens"), list) else []
        for ti, tok in enumerate(toks):
            if not isinstance(tok, dict):
                continue
            word = str(tok.get("word", "")).strip()
            if not word:
                continue
            rows.append(
                {
                    "segment": si,
                    "token_i": ti,
                    "word": word,
                    "start_s": _f(tok.get("start_s")),
                    "end_s": _f(tok.get("end_s")),
                    "confidence": _f(tok.get("confidence")),
                }
            )

    # fallback: transcript.meta.words
    if not rows:
        meta = transcript.get("meta") if isinstance(transcript.get("meta"), dict) else {}
        words = meta.get("words") if isinstance(meta.get("words"), list) else []
        for wi, w in enumerate(words):
            if not isinstance(w, dict):
                continue
            word = str(w.get("word", "")).strip()
            if not word:
                continue
            rows.append(
                {
                    "segment": None,
                    "token_i": wi,
                    "word": word,
                    "start_s": _f(w.get("start_s")),
                    "end_s": _f(w.get("end_s")),
                    "confidence": _f(w.get("confidence")),
                }
            )

    return rows


def _render_error_heatmap_if_present(run_dir: str) -> None:
    """
    Draws heatmap from artifacts/timeline.json when it exists.
    If not present, explains why.
    """
    art_timeline = os.path.join(run_dir, "artifacts", "timeline.json")
    if not os.path.exists(art_timeline):
        st.info(
            "No error heatmap found (missing artifacts/timeline.json). "
            "This file is only produced when word-level attribution is available "
            "(Reference text + word attribution enabled + successful CTC alignment)."
        )
        return

    try:
        data = read_json(art_timeline)
    except Exception as e:
        st.warning(f"Could not read timeline.json: {e}")
        return

    bins = data.get("bins") if isinstance(data, dict) else None
    if not isinstance(bins, list) or not bins:
        st.info("timeline.json exists but has no bins.")
        return

    rows = []
    for i, b in enumerate(bins):
        if not isinstance(b, dict):
            continue

        # common schemas we might have
        start_s = _f(b.get("start_s", b.get("t0_s")))
        end_s = _f(b.get("end_s", b.get("t1_s")))
        if start_s is None and "window" in b:
            start_s, end_s = _parse_window_to_start_end(b.get("window"))

        sub = int(b.get("sub", b.get("substitutions", 0)) or 0)
        ins = int(b.get("ins", b.get("insertions", 0)) or 0)
        dele = int(b.get("del", b.get("deletions", 0)) or 0)

        total = b.get("total")
        if total is None:
            total = sub + ins + dele
        total = int(total or 0)

        rows.append(
            {
                "bin": i,
                "start_s": start_s if start_s is not None else float(i),
                "end_s": end_s if end_s is not None else float(i + 1),
                "sub": sub,
                "ins": ins,
                "del": dele,
                "total": total,
            }
        )

    if not rows:
        st.info("No usable bins parsed from timeline.json.")
        return

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df_long = df.melt(
            id_vars=["bin", "start_s", "end_s"],
            value_vars=["sub", "ins", "del"],
            var_name="type",
            value_name="count",
        )

        st.subheader("Error heatmap (sub/ins/del by time bin)")
        try:
            import altair as alt  # type: ignore

            chart = (
                alt.Chart(df_long)
                .mark_rect()
                .encode(
                    x=alt.X("start_s:Q", title="time (s)"),
                    y=alt.Y("type:N", title="error type"),
                    color=alt.Color("count:Q", title="count"),
                    tooltip=["start_s:Q", "end_s:Q", "type:N", "count:Q"],
                )
                .properties(height=110)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            # fallback: show bars
            st.bar_chart(df.set_index("start_s")[["sub", "ins", "del"]])

        with st.expander("timeline.json (raw)", expanded=False):
            st.json(data)
    except Exception as e:
        st.warning(f"Could not render error heatmap: {e}")


def _render_confidence_heatmap(report: dict[str, Any]) -> None:
    """
    Always available when backend yields word tokens (faster-whisper does).
    This gives a 'heatmap' of mean confidence over time (bin_s).
    """
    words = _extract_word_tokens(report)
    if not words:
        st.info("No word tokens found to build a confidence heatmap.")
        return

    duration_s = _f(report.get("audio_duration_s"))
    if duration_s is None:
        ends = [w.get("end_s") for w in words if w.get("end_s") is not None]
        duration_s = max(ends) if ends else 0.0

    bin_s = float(st.selectbox("Confidence heatmap bin (seconds)", [0.25, 0.5, 1.0, 2.0], index=2))
    n_bins = int(max(1, (duration_s / bin_s) + 1))

    sums = [0.0] * n_bins
    cnts = [0] * n_bins
    mins = [1.0] * n_bins

    for w in words:
        s = w.get("start_s") if w.get("start_s") is not None else 0.0
        e = w.get("end_s") if w.get("end_s") is not None else s
        c = w.get("confidence")
        if c is None:
            continue
        mid = 0.5 * (float(s) + float(e))
        bi = int(mid // bin_s)
        bi = max(0, min(n_bins - 1, bi))
        sums[bi] += float(c)
        cnts[bi] += 1
        mins[bi] = min(float(mins[bi]), float(c))

    rows = []
    for i in range(n_bins):
        start = i * bin_s
        end = (i + 1) * bin_s
        mean = (sums[i] / cnts[i]) if cnts[i] > 0 else None
        mmin = mins[i] if cnts[i] > 0 else None
        rows.append({"bin": i, "start_s": start, "end_s": end, "mean_conf": mean, "min_conf": mmin, "n_words": cnts[i]})

    st.subheader("Confidence heatmap (mean/min confidence by time bin)")
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df_plot = df.dropna(subset=["mean_conf"]).copy()

        try:
            import altair as alt  # type: ignore

            chart = (
                alt.Chart(df_plot)
                .mark_rect()
                .encode(
                    x=alt.X("start_s:Q", title="time (s)"),
                    y=alt.value("confidence"),  # 1-row heatmap
                    color=alt.Color("mean_conf:Q", title="mean conf"),
                    tooltip=["start_s:Q", "end_s:Q", "mean_conf:Q", "min_conf:Q", "n_words:Q"],
                )
                .properties(height=60)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.line_chart(df.set_index("start_s")[["mean_conf"]])

        with st.expander("Confidence bins (table)", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Could not render confidence heatmap: {e}")


def _render_tokens_confidence_and_meta(run_dir: str) -> None:
    rp = os.path.join(run_dir, "report.json")
    if not os.path.exists(rp):
        return

    try:
        report = read_json(rp)
    except Exception:
        return

    transcript = report.get("transcript") if isinstance(report, dict) else None
    if not isinstance(transcript, dict):
        return

    with st.expander("🔎 Tokens / confidence / backend meta", expanded=False):
        # Show meta.info if present (language probs, decoding options, etc.)
        meta = transcript.get("meta") if isinstance(transcript.get("meta"), dict) else {}
        info = meta.get("info") if isinstance(meta.get("info"), dict) else {}
        if info:
            st.subheader("Transcript meta.info")
            st.json(info)

        # Error heatmap if present, else explanation
        _render_error_heatmap_if_present(run_dir)

        # Always provide a confidence heatmap fallback
        _render_confidence_heatmap(report)

        # Token table
        rows = _extract_word_tokens(report)
        if not rows:
            st.info("No tokens found in transcript.segments[*].tokens or transcript.meta.words.")
            return

        # time/conf filters
        duration_s = _f(report.get("audio_duration_s"))
        if duration_s is None:
            ends = [r.get("end_s") for r in rows if r.get("end_s") is not None]
            duration_s = max(ends) if ends else 1.0

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            t0, t1 = st.slider(
                "Token time range (s)",
                min_value=0.0,
                max_value=float(duration_s) if duration_s else 1.0,
                value=(0.0, float(duration_s) if duration_s else 1.0),
                step=0.1,
            )
        with c2:
            min_conf = st.slider("Min token confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        with c3:
            show_lowest = st.checkbox("Show 20 lowest-confidence words", value=True)

        filt = []
        for r in rows:
            s = r.get("start_s") if r.get("start_s") is not None else 0.0
            e = r.get("end_s") if r.get("end_s") is not None else s
            conf = r.get("confidence")
            if e < t0 or s > t1:
                continue
            if conf is not None and conf < float(min_conf):
                continue
            filt.append(r)

        confs = [r["confidence"] for r in filt if r.get("confidence") is not None]
        if confs:
            confs_sorted = sorted(confs)
            mean_conf = sum(confs_sorted) / len(confs_sorted)
            p10 = confs_sorted[int(0.10 * (len(confs_sorted) - 1))] if len(confs_sorted) > 1 else confs_sorted[0]
            p50 = confs_sorted[int(0.50 * (len(confs_sorted) - 1))] if len(confs_sorted) > 1 else confs_sorted[0]
            st.caption(f"Token confidence: mean={mean_conf:.3f} | p10={p10:.3f} | p50={p50:.3f} | min={confs_sorted[0]:.3f}")

        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(filt)
            st.subheader("Word tokens")
            st.dataframe(df, use_container_width=True, hide_index=True)

            if show_lowest and "confidence" in df.columns:
                low = df.dropna(subset=["confidence"]).sort_values("confidence", ascending=True).head(20)
                if len(low) > 0:
                    st.subheader("Lowest-confidence words (top 20)")
                    st.dataframe(low, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(filt, use_container_width=True)


def _page_single_run(base_runs_dir: str) -> None:
    st.header("Single Run (Analyze)")

    runs = list_run_dirs(base_runs_dir)
    with st.expander("Open existing run", expanded=bool(runs)):
        choice = st.selectbox("Select run", options=["(none)"] + [human_relpath(r, base_runs_dir) for r in runs])
        if choice != "(none)":
            run_dir = os.path.join(base_runs_dir, choice)
            render_run_report(run_dir)
            _render_tokens_confidence_and_meta(run_dir)  # ✅ added
            st.divider()

    st.subheader("New analysis")

    audio_path = _audio_input(base_runs_dir)
    if not audio_path:
        st.info("Upload an audio file or provide a server path to continue.")
        return

    _eng, _reg, backend_names = get_engine_and_registry()
    backend_name = st.selectbox("Backend", options=backend_names, index=0)
    backend_params = _backend_param_editor(backend_name)
    ref_text = _reference_editor()

    with st.expander("Alignment + Attribution", expanded=True):
        word_align = st.checkbox("Enable word alignment (CTC)", value=bool(ref_text))
        word_attribution = st.checkbox("Enable word-level attribution (events + moments)", value=bool(ref_text))
        aligner_model = st.text_input("CTC aligner model", value="facebook/wav2vec2-base-960h")
        aligner_device = st.selectbox("Aligner device", ["cpu", "cuda"], index=0)

    with st.expander("Streaming simulation / true streaming", expanded=False):
        streaming_enabled = st.checkbox("Enable streaming mode", value=False)
        chunk_ms = int(st.number_input("chunk_ms", min_value=80, max_value=5000, value=800, step=20))
        overlap_ms = int(st.number_input("overlap_ms", min_value=0, max_value=2000, value=160, step=10))
        right_context_ms = int(st.number_input("right_context_ms (simulated)", min_value=0, max_value=2000, value=0, step=10))
        repeats = int(st.number_input("repeats (for p95 estimation)", min_value=1, max_value=50, value=5, step=1))

    with st.expander("Profiling / Tracing", expanded=False):
        enable_profiling = st.checkbox("Enable stage profiling (backend profiler)", value=True)
        torch_trace = st.checkbox("Export torch profiler trace (best-effort)", value=False)

    run_name = st.text_input("Run name (optional)", value="")
    run_id = slugify(run_name) if run_name.strip() else now_run_id("run")
    out_dir = os.path.join(base_runs_dir, run_id)
    st.code(f"Output dir: {out_dir}", language="text")

    if st.button("🚀 Run analysis", type="primary", use_container_width=True):
        ensure_dir(out_dir)

        # snapshot audio into run dir (nice for reproducibility)
        audio_to_use = audio_path
        try:
            import shutil

            bn = os.path.basename(audio_path)
            snap = os.path.join(out_dir, bn)
            if os.path.exists(audio_path) and os.path.abspath(audio_path) != os.path.abspath(snap):
                shutil.copy2(audio_path, snap)
                audio_to_use = snap
        except Exception:
            pass

        with st.status("Running analysis...", expanded=True) as status:
            try:
                run_single_analysis(
                    audio_path=audio_to_use,
                    out_dir=out_dir,
                    backend_name=backend_name,
                    backend_params=backend_params,
                    reference_text=ref_text,
                    repeats=repeats,
                    word_align=word_align,
                    aligner_model=aligner_model,
                    aligner_device=aligner_device,
                    word_attribution=word_attribution,
                    streaming_enabled=streaming_enabled,
                    chunk_ms=chunk_ms,
                    overlap_ms=overlap_ms,
                    right_context_ms=right_context_ms,
                    torch_trace=torch_trace,
                    enable_profiling=enable_profiling,
                )
                status.update(label="Done ✅", state="complete")
            except Exception as e:
                status.update(label="Failed ❌", state="error")
                st.exception(e)
                return

        st.success(f"Run completed: {out_dir}")
        render_run_report(out_dir)
        _render_tokens_confidence_and_meta(out_dir)  # ✅ added


def _page_dataset(base_runs_dir: str) -> None:
    st.header("Dataset Runner (Batch)")
    st.caption("Manifest CSV needs `audio_path` and optionally `ref_text` or `ref_path` + metadata columns.")

    up = st.file_uploader("Upload manifest CSV", type=["csv"])
    if not up:
        st.info("Upload a CSV manifest to continue.")
        return

    run_id = now_run_id("dataset")
    out_dir = os.path.join(base_runs_dir, run_id)
    ensure_dir(out_dir)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "wb") as f:
        f.write(up.getbuffer())

    _eng, _reg, backend_names = get_engine_and_registry()
    backend_name = st.selectbox("Backend", options=backend_names, index=0)
    backend_params = _backend_param_editor(backend_name)

    with st.expander("Alignment + Attribution", expanded=False):
        word_align = st.checkbox("Enable word alignment (CTC)", value=False)
        word_attribution = st.checkbox("Enable word-level attribution (events + moments)", value=True)
        aligner_model = st.text_input("CTC aligner model", value="facebook/wav2vec2-base-960h")
        aligner_device = st.selectbox("Aligner device", ["cpu", "cuda"], index=0)

    with st.expander("Streaming", expanded=False):
        streaming_enabled = st.checkbox("Enable streaming mode", value=False)
        chunk_ms = int(st.number_input("chunk_ms", min_value=80, max_value=5000, value=800, step=20))
        overlap_ms = int(st.number_input("overlap_ms", min_value=0, max_value=2000, value=160, step=10))
        right_context_ms = int(st.number_input("right_context_ms", min_value=0, max_value=2000, value=0, step=10))
        repeats = int(st.number_input("repeats", min_value=1, max_value=20, value=1, step=1))

    st.code(f"Output dir: {out_dir}", language="text")

    prog = st.progress(0.0)
    msg = st.empty()

    def progress(frac: float, text: str) -> None:
        prog.progress(max(0.0, min(1.0, float(frac))))
        msg.info(text)

    if st.button("🚀 Run dataset", type="primary", use_container_width=True):
        with st.status("Running dataset...", expanded=True) as status:
            try:
                run_dataset_web(
                    manifest_path=manifest_path,
                    out_dir=out_dir,
                    backend_name=backend_name,
                    backend_params=backend_params,
                    repeats=repeats,
                    word_align=word_align,
                    aligner_model=aligner_model,
                    aligner_device=aligner_device,
                    word_attribution=word_attribution,
                    streaming_enabled=streaming_enabled,
                    chunk_ms=chunk_ms,
                    overlap_ms=overlap_ms,
                    right_context_ms=right_context_ms,
                    progress=progress,
                )
                status.update(label="Done ✅", state="complete")
            except Exception as e:
                status.update(label="Failed ❌", state="error")
                st.exception(e)
                return

        st.success(f"Dataset run completed: {out_dir}")
        render_dataset_report(out_dir)

    st.divider()
    st.subheader("Open existing dataset run")
    runs = [r for r in list_run_dirs(base_runs_dir) if os.path.exists(os.path.join(r, "dataset_summary.json"))]
    if runs:
        choice = st.selectbox("Select dataset run", options=[human_relpath(r, base_runs_dir) for r in runs])
        render_dataset_report(os.path.join(base_runs_dir, choice))
    else:
        st.info("No previous dataset runs found.")


def _page_compare(base_runs_dir: str) -> None:
    st.header("Compare Runs")

    runs = [r for r in list_run_dirs(base_runs_dir) if os.path.exists(os.path.join(r, "report.json"))]
    if len(runs) < 2:
        st.info("Need at least 2 single runs with report.json to compare.")
        return

    left = st.selectbox("Run A", options=[human_relpath(r, base_runs_dir) for r in runs], index=0)
    right = st.selectbox("Run B", options=[human_relpath(r, base_runs_dir) for r in runs], index=min(1, len(runs) - 1))

    a_dir = os.path.join(base_runs_dir, left)
    b_dir = os.path.join(base_runs_dir, right)

    ra = read_json(os.path.join(a_dir, "report.json"))
    rb = read_json(os.path.join(b_dir, "report.json"))

    def get_metric(r: dict, path: list[str]) -> float | None:
        cur: Any = r
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        try:
            return float(cur) if cur is not None else None
        except Exception:
            return None

    st.subheader("Key deltas (B - A)")
    rows = [
        ("WER", get_metric(ra, ["wer", "wer"]), get_metric(rb, ["wer", "wer"])),
        ("Latency p95", get_metric(ra, ["latency", "percentiles_s", "p95"]), get_metric(rb, ["latency", "percentiles_s", "p95"])),
        ("FW latency p95", get_metric(ra, ["streaming", "first_word_latency_percentiles_s", "p95"]), get_metric(rb, ["streaming", "first_word_latency_percentiles_s", "p95"])),
    ]

    for name, va, vb in rows:
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            st.write(name)
        with c2:
            st.write("A:", va if va is not None else "n/a")
        with c3:
            st.write("B:", vb if vb is not None else "n/a")
        if va is not None and vb is not None:
            st.caption(f"Δ (B-A) = {vb - va:+.6f}")


def _page_validate() -> None:
    st.header("Backend Validator")

    _, _, backend_names = get_engine_and_registry()
    backend_name = st.selectbox("Backend", options=backend_names, index=0)

    if st.button("Run contract tests", type="primary", use_container_width=True):
        with st.status("Validating backend...", expanded=True) as status:
            try:
                res = validate_backend_web(backend_name)
                status.update(label="Done ✅", state="complete")
            except Exception as e:
                status.update(label="Failed ❌", state="error")
                st.exception(e)
                return
        st.subheader("Results")
        st.json(res)


def _page_about() -> None:
    st.header("About")
    st.markdown(
        """
ASR-LE Web App provides **advanced ASR evaluation** beyond CLI:

- Word-level error attribution with timestamps
- Error timeline heatmaps
- Auto top-3 “error moments” (timestamp-only clips)
- Streaming p95 first-word latency estimation
- Dataset batch runner
- Backend contract tests and streaming interface validation
"""
    )


def main() -> None:
    st.set_page_config(page_title="ASR-LE Web App", layout="wide")

    base_runs_dir_default = "runs"
    globals_ = _sidebar_global(base_runs_dir_default)  # ✅ fixed
    base_runs_dir = globals_["base_runs_dir"]

    page = st.sidebar.radio(
        "Navigation",
        ["Single Run", "Dataset Runner", "Compare Runs", "Backend Validator", "About"],
        index=0,
    )

    if page == "Single Run":
        _page_single_run(base_runs_dir)
    elif page == "Dataset Runner":
        _page_dataset(base_runs_dir)
    elif page == "Compare Runs":
        _page_compare(base_runs_dir)
    elif page == "Backend Validator":
        _page_validate()
    else:
        _page_about()


if __name__ == "__main__":
    main()
