from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asrle.backends.base import ASRBackend
from asrle.core.profiler import Profiler
from asrle.types import Segment, Transcript


@dataclass
class GenericAPIBackend(ASRBackend):
    """
    Generic HTTP backend.
    Expects API to return:
      { "text": "...", "segments": [{"start_s":..,"end_s":..,"text":"..."}], "language": "..." }
    Requires: asr-le[api]
    """
    url: str = ""
    timeout_s: float = 60.0
    headers: dict[str, str] | None = None

    @classmethod
    def backend_name(cls) -> str:
        return "api"

    def transcribe(self, audio_path: str, profiler: Profiler | None = None) -> Transcript:
        profiler = profiler or Profiler()
        try:
            import requests  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing requests. Install: pip install -e '.[api]'") from e

        if not self.url:
            raise ValueError("api backend requires --url")

        def call():
            with open(audio_path, "rb") as f:
                files = {"audio": f}
                r = requests.post(self.url, files=files, headers=self.headers or {}, timeout=self.timeout_s)
            r.raise_for_status()
            return r.json()

        out: dict[str, Any] = profiler.time("api_call", call)

        text = str(out.get("text", "")).strip()
        language = out.get("language")
        segs = []
        for s in out.get("segments", []) or []:
            segs.append(Segment(float(s.get("start_s", 0.0)), float(s.get("end_s", 0.0)), str(s.get("text", "")).strip()))
        if not segs:
            segs = [Segment(0.0, 0.0, text)]

        return Transcript(text=text, segments=segs, language=language, meta={"raw": out})
