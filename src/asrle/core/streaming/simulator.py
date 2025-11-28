from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from asrle.backends.base import ASRBackend
from asrle.backends.streaming import StreamingASRBackend
from asrle.core.profiler import Profiler
from asrle.types import Segment, StreamingConfig, StreamingReport, Transcript
from asrle.utils.audio import load_audio_mono, resample, slice_to_wav
from asrle.utils.text import normalize_text
from asrle.utils.time import Timer


def _dedup_words(prev_text: str, new_text: str, max_ngram: int = 12) -> str:
    p = normalize_text(prev_text).split()
    n = normalize_text(new_text).split()
    if not p:
        return new_text.strip()
    if not n:
        return prev_text.strip()

    best_k = 0
    for k in range(1, min(max_ngram, len(p), len(n)) + 1):
        if p[-k:] == n[:k]:
            best_k = k

    if best_k == 0:
        return (prev_text.strip() + " " + new_text.strip()).strip()

    merged_words = p + n[best_k:]
    return " ".join(merged_words).strip()


def _first_word_latency_from_progress(progress_texts: list[str], wall_s: list[float]) -> float | None:
    acc = 0.0
    for txt, w in zip(progress_texts, wall_s):
        acc += float(w)
        if normalize_text(txt).strip():
            return float(acc)
    return None


@dataclass
class StreamingSimulator:
    cfg: StreamingConfig

    def simulate(self, backend: ASRBackend, audio_path: str) -> StreamingReport:
        chunk_s = self.cfg.chunk_ms / 1000.0
        overlap_s = self.cfg.overlap_ms / 1000.0
        right_s = self.cfg.right_context_ms / 1000.0

        per_chunk_wall: list[float] = []
        merge_notes: list[str] = []
        progress_texts: list[str] = []

        merged_text = ""
        merged_segments: list[Segment] = []

        # TRUE STREAMING PATH
        if isinstance(backend, StreamingASRBackend) and backend.supports_streaming():
            # In true streaming, "overlap/right_context" is typically a model-internal concern.
            # We keep overlap by duplicating a short tail into the next push (best-effort simulation).
            audio, sr = load_audio_mono(audio_path)
            audio, sr = resample(audio, sr, 16000)
            sr = 16000

            session = backend.create_stream(sample_rate=sr, profiler=None)

            step = int(round(chunk_s * sr))
            ovlp = int(round(overlap_s * sr))
            tail = np.zeros((0,), dtype=np.float32)

            idx = 0
            chunk_idx = 0
            while idx < len(audio):
                if self.cfg.max_chunks is not None and chunk_idx >= self.cfg.max_chunks:
                    merge_notes.append("Stopped due to max_chunks limit.")
                    break

                chunk = audio[idx : idx + step]
                idx += step

                # overlap simulation: prepend tail (duplicate audio)
                if ovlp > 0 and tail.size > 0:
                    chunk_in = np.concatenate([tail, chunk], axis=0)
                else:
                    chunk_in = chunk

                # right context in true streaming is not the same thing. Mention explicitly.
                if right_s > 0:
                    merge_notes.append("right_context_ms is ignored in true streaming mode (backend controls lookahead).")

                prof = Profiler()
                with Timer("stream_push") as tt:
                    tr = session.push_audio(chunk_in.astype(np.float32), sr, is_final=False, profiler=prof)
                per_chunk_wall.append(tt.elapsed_s)

                progress_texts.append(tr.text or "")

                # merge text (some streaming backends revise earlier output; we treat as latest best effort)
                if not merged_text:
                    merged_text = (tr.text or "").strip()
                else:
                    # if backend returns full transcript each time, replacing is better than ngram merge.
                    # heuristic: if new text length >= old text length, accept it; else merge.
                    if len((tr.text or "").strip()) >= len(merged_text):
                        merged_text = (tr.text or "").strip()
                    else:
                        merged_text = _dedup_words(merged_text, tr.text or "")

                # segments: shift unknown; assume backend already uses relative stream times or none.
                merged_segments = tr.segments or merged_segments

                # update tail for overlap duplication
                if ovlp > 0:
                    tail = chunk[-ovlp:] if chunk.size >= ovlp else chunk

                chunk_idx += 1

            # final flush
            prof = Profiler()
            with Timer("stream_final") as tt:
                tr = session.push_audio(np.zeros((0,), dtype=np.float32), sr, is_final=True, profiler=prof)
            per_chunk_wall.append(tt.elapsed_s)
            progress_texts.append(tr.text or "")
            if len((tr.text or "").strip()) >= len(merged_text):
                merged_text = (tr.text or "").strip()

            fwlat = _first_word_latency_from_progress(progress_texts, per_chunk_wall)

            return StreamingReport(
                config=self.cfg,
                transcript=Transcript(text=merged_text.strip(), segments=merged_segments, meta={"mode": "true-streaming"}),
                wall_clock_total_s=float(sum(per_chunk_wall)),
                per_chunk_wall_s=[float(x) for x in per_chunk_wall],
                first_word_latency_s=fwlat,
                merge_notes=merge_notes,
            )

        # FALLBACK FILE-CHUNK SIMULATION PATH
        # (works for any backend; uses slicing into WAV chunks)
        from asrle.utils.audio import probe_audio

        info = probe_audio(audio_path)
        if info.duration_s is None:
            raise RuntimeError("Streaming simulation requires known audio duration. Use WAV/FLAC or install codecs.")

        t = 0.0
        chunk_idx = 0
        total_wall = 0.0

        while t < info.duration_s:
            if self.cfg.max_chunks is not None and chunk_idx >= self.cfg.max_chunks:
                merge_notes.append("Stopped due to max_chunks limit.")
                break

            decode_start = max(0.0, t - overlap_s)
            decode_end = min(info.duration_s, t + chunk_s + right_s)

            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, f"chunk_{chunk_idx}.wav")
                actual_s, _actual_e = slice_to_wav(audio_path, decode_start, decode_end, wav_path, target_sr=16000)

                prof = Profiler()
                with Timer("chunk_total") as tt:
                    tr = backend.transcribe(wav_path, profiler=prof)
                per_chunk_wall.append(tt.elapsed_s)
                total_wall += tt.elapsed_s

                progress_texts.append(tr.text or "")

                # shift + trim emitted segments
                shifted: list[Segment] = []
                cutoff = t + chunk_s
                for s in tr.segments:
                    ss = s.start_s + actual_s
                    ee = s.end_s + actual_s
                    if right_s > 0:
                        if ss >= cutoff:
                            continue
                        ee = min(ee, cutoff)
                    if ee <= ss:
                        continue
                    shifted.append(Segment(ss, ee, s.text, tokens=s.tokens))

                if merged_text:
                    merged_text = _dedup_words(merged_text, tr.text or "")
                else:
                    merged_text = (tr.text or "").strip()

                merged_segments.extend(shifted)

            t += chunk_s
            chunk_idx += 1

        merged_segments = sorted(merged_segments, key=lambda s: (s.start_s, s.end_s))
        fwlat = _first_word_latency_from_progress(progress_texts, per_chunk_wall)

        return StreamingReport(
            config=self.cfg,
            transcript=Transcript(text=merged_text.strip(), segments=merged_segments, meta={"mode": "simulated-streaming"}),
            wall_clock_total_s=float(total_wall),
            per_chunk_wall_s=[float(x) for x in per_chunk_wall],
            first_word_latency_s=fwlat,
            merge_notes=merge_notes,
        )
