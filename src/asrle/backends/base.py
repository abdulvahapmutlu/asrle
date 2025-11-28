from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from asrle.core.profiler import Profiler
from asrle.types import Transcript


class ASRBackend(ABC):
    """
    Backends should keep transcribe() stable:
    - accept audio_path
    - return Transcript with text + segments (timestamps if available)
    - optionally use profiler.time(stage, fn) for stage breakdowns

    Optional streaming capability:
    - implement create_stream(sample_rate, profiler) -> ASRStreamSession (see asrle.backends.streaming)
    """

    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: str, profiler: Profiler | None = None) -> Transcript:
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        return callable(getattr(self, "create_stream", None))

    def info(self) -> dict[str, Any]:
        return {"name": self.backend_name(), "supports_streaming": self.supports_streaming()}
