from __future__ import annotations

from asrle.backends.base import ASRBackend
from asrle.core.profiler import Profiler


def validate_backend_contract(backend: ASRBackend, audio_path: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    prof = Profiler()

    try:
        tr = backend.transcribe(audio_path, profiler=prof)
    except Exception as e:
        return False, [f"transcribe() raised: {e}"]

    if tr is None:
        return False, ["transcribe() returned None"]

    if not isinstance(tr.text, str):
        issues.append("Transcript.text must be str")

    if not isinstance(tr.segments, list):
        issues.append("Transcript.segments must be a list")

    if tr.segments:
        # monotonic + valid
        last_end = -1e9
        for idx, s in enumerate(tr.segments):
            if s.end_s < s.start_s:
                issues.append(f"Segment[{idx}] end_s < start_s ({s.end_s} < {s.start_s})")
            if s.start_s < last_end - 0.1:
                issues.append(f"Segment[{idx}] is not monotonic (start_s={s.start_s} < prev_end={last_end})")
            last_end = max(last_end, s.end_s)

            if not isinstance(s.text, str):
                issues.append(f"Segment[{idx}].text must be str")

            # token contract if present
            for t in (s.tokens or []):
                if "word" in t:
                    if ("start_s" not in t and "start" not in t) or ("end_s" not in t and "end" not in t):
                        issues.append(f"Segment[{idx}] token with 'word' must include start/end timestamps")
    else:
        # allow empty segments, but text should still exist
        if tr.text.strip() == "":
            issues.append("Empty transcript: provide text or segments (both empty).")

    # profiler sanity if provided: not required, but shouldn’t contain nonsense
    for k, v in prof.stages_s.items():
        if v < 0:
            issues.append(f"Profiler stage '{k}' has negative time: {v}")

    return (len(issues) == 0), issues
