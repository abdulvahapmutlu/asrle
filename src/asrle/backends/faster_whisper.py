from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asrle.backends.base import ASRBackend
from asrle.core.profiler import Profiler
from asrle.types import Segment, Transcript


@dataclass
class FasterWhisperBackend(ASRBackend):
    """
    faster-whisper wrapper.
    Requires: asr-le[faster_whisper]
    """
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: str | None = None
    vad_filter: bool = False

    _model: Any = field(default=None, init=False, repr=False)

    @classmethod
    def backend_name(cls) -> str:
        return "faster-whisper"

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise RuntimeError("Missing faster-whisper. Install: pip install -e '.[faster_whisper]'") from e
        return WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def _get_model(self, profiler: Profiler) -> Any:
        if self._model is None:
            self._model = profiler.time("init", self._load_model)
        return self._model

    def transcribe(self, audio_path: str, profiler: Profiler | None = None) -> Transcript:
        profiler = profiler or Profiler()
        model = self._get_model(profiler)

        segments_iter, info = profiler.time(
            "decode",
            lambda: model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                language=self.language,
                vad_filter=self.vad_filter,
                word_timestamps=True,
            ),
        )

        segments: list[Segment] = []
        full_text_parts: list[str] = []
        words_meta: list[dict[str, Any]] = []

        for s in segments_iter:
            full_text_parts.append(s.text)
            tokens: list[dict[str, Any]] = []
            if getattr(s, "words", None):
                for w in s.words:
                    tok = {
                        "word": str(w.word).strip(),
                        "start_s": float(w.start),
                        "end_s": float(w.end),
                        "confidence": float(getattr(w, "probability", 0.0))
                        if getattr(w, "probability", None) is not None
                        else None,
                    }
                    tokens.append(tok)
                    words_meta.append({**tok, "source": "backend"})
            segments.append(Segment(float(s.start), float(s.end), s.text.strip(), tokens=tokens))

        text = " ".join(full_text_parts).strip()
        meta: dict[str, Any] = {"info": getattr(info, "__dict__", {})}
        if words_meta:
            meta["words"] = words_meta
        return Transcript(text=text, segments=segments, language=self.language, meta=meta)
