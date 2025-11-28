from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from asrle.core.profiler import Profiler
from asrle.types import Transcript


@runtime_checkable
class ASRStreamSession(Protocol):
    """
    True streaming/incremental decoding session.

    Contract:
    - push_audio() can be called many times with sequential audio chunks.
    - returns partial Transcript each time (may revise earlier text OR be incremental).
    - if is_final=True, backend should flush any buffered hypothesis.
    """

    def push_audio(
        self,
        audio: np.ndarray,
        sr: int,
        is_final: bool = False,
        profiler: Profiler | None = None,
    ) -> Transcript: ...

    def reset(self) -> None: ...


@runtime_checkable
class StreamingASRBackend(Protocol):
    """
    Optional capability: a backend may implement create_stream().
    """

    def create_stream(
        self,
        sample_rate: int = 16000,
        profiler: Profiler | None = None,
    ) -> ASRStreamSession: ...
