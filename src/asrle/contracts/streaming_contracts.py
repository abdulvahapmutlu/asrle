from __future__ import annotations

import numpy as np

from asrle.backends.base import ASRBackend
from asrle.backends.streaming import StreamingASRBackend
from asrle.core.profiler import Profiler


def validate_streaming_contract(backend: ASRBackend) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not (isinstance(backend, StreamingASRBackend) and backend.supports_streaming()):
        return True, ["Streaming not supported by this backend (skipped)."]

    try:
        sess = backend.create_stream(sample_rate=16000, profiler=None)
    except Exception as e:
        return False, [f"create_stream() raised: {e}"]

    # feed 4 chunks of 250ms silence
    prof = Profiler()
    try:
        for _ in range(4):
            chunk = np.zeros((4000,), dtype=np.float32)  # 0.25s @ 16k
            tr = sess.push_audio(chunk, 16000, is_final=False, profiler=prof)
            if tr is None or not isinstance(tr.text, str):
                issues.append("push_audio() must return Transcript with .text str")
        tr = sess.push_audio(np.zeros((0,), dtype=np.float32), 16000, is_final=True, profiler=prof)
        if tr is None or not isinstance(tr.text, str):
            issues.append("final push_audio() must return Transcript")
    except Exception as e:
        return False, [f"push_audio() raised: {e}"]

    return (len([x for x in issues if "skipped" not in x.lower()]) == 0), issues
