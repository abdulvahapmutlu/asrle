from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asrle.backends.base import ASRBackend
from asrle.core.profiler import Profiler
from asrle.types import Segment, Transcript


@dataclass
class HuggingFaceWhisperBackend(ASRBackend):
    """
    Transformers ASR pipeline wrapper.
    Requires: asr-le[hf]
    """
    model_name: str = "openai/whisper-small"
    device: str = "cpu"  # "cpu" or "cuda"
    language: str | None = None
    task: str = "transcribe"
    chunk_length_s: float | None = None

    _pipe: Any = field(default=None, init=False, repr=False)

    @classmethod
    def backend_name(cls) -> str:
        return "hf-whisper"

    def _load_pipe(self):
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("Missing transformers. Install: pip install -e '.[hf]'") from e

        device_id = 0 if self.device == "cuda" else -1
        return pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=device_id,
            generate_kwargs={"task": self.task, **({"language": self.language} if self.language else {})},
            chunk_length_s=self.chunk_length_s,
            return_timestamps="word",
        )

    def _get_pipe(self, profiler: Profiler) -> Any:
        if self._pipe is None:
            self._pipe = profiler.time("init", self._load_pipe)
        return self._pipe

    def transcribe(self, audio_path: str, profiler: Profiler | None = None) -> Transcript:
        profiler = profiler or Profiler()
        pipe = self._get_pipe(profiler)

        out = profiler.time("decode", lambda: pipe(audio_path))
        text = (out.get("text") or "").strip()

        segments: list[Segment] = []
        words_meta: list[dict[str, Any]] = []

        chunks = out.get("chunks") or out.get("segments") or []
        if chunks:
            for c in chunks:
                ts = c.get("timestamp") or c.get("timestamps")
                if ts and isinstance(ts, (tuple, list)) and len(ts) == 2:
                    start, end = ts
                    start_s = float(start or 0.0)
                    end_s = float(end or start_s)
                else:
                    start_s, end_s = 0.0, 0.0

                wtxt = str(c.get("text", "")).strip()
                token = {"word": wtxt, "start_s": start_s, "end_s": end_s, "confidence": None}
                words_meta.append({**token, "source": "backend"})
                segments.append(Segment(start_s, end_s, wtxt, tokens=[token]))
        else:
            segments.append(Segment(0.0, 0.0, text))

        meta: dict[str, Any] = {"raw": {k: v for k, v in out.items() if k != "chunks"}}
        if words_meta:
            meta["words"] = words_meta
        return Transcript(text=text, segments=segments, language=self.language, meta=meta)
