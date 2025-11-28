from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from asrle.backends.base import ASRBackend
from asrle.backends.streaming import ASRStreamSession
from asrle.core.profiler import Profiler
from asrle.types import Segment, Transcript


class _DummyStream(ASRStreamSession):
    def __init__(self, text: str = "hello world", emit_after_s: float = 0.25):
        self._text = text
        self._emit_after_s = emit_after_s
        self._seen_s = 0.0
        self._emitted = False

    def reset(self) -> None:
        self._seen_s = 0.0
        self._emitted = False

    def push_audio(
        self, audio: np.ndarray, sr: int, is_final: bool = False, profiler: Profiler | None = None
    ) -> Transcript:
        profiler = profiler or Profiler()
        dur = float(len(audio) / max(1, sr))
        self._seen_s += dur

        # simulate "first word appears after some audio seen"
        if not self._emitted and (self._seen_s >= self._emit_after_s or is_final):
            self._emitted = True

        txt = self._text if self._emitted else ""
        segs = [Segment(0.0, max(0.01, self._seen_s), txt)] if txt else []
        return Transcript(text=txt, segments=segs, language="en", meta={"streaming": True})


@dataclass
class DummyBackend(ASRBackend):
    """
    Deterministic backend for tests/demos.
    Now also supports a true streaming session.
    """
    text: str = "hello world"
    seconds: float = 2.0

    @classmethod
    def backend_name(cls) -> str:
        return "dummy"

    def transcribe(self, audio_path: str, profiler: Profiler | None = None) -> Transcript:
        profiler = profiler or Profiler()
        profiler.time("feature", lambda: None)
        profiler.time("encoder", lambda: None)
        profiler.time("decoder", lambda: None)
        seg = Segment(0.0, self.seconds, self.text)
        return Transcript(text=self.text, segments=[seg], language="en", meta={"audio_path": audio_path})

    def create_stream(self, sample_rate: int = 16000, profiler: Profiler | None = None) -> ASRStreamSession:
        _ = sample_rate
        _ = profiler
        return _DummyStream(text=self.text, emit_after_s=0.25)
