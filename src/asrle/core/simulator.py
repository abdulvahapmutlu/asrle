from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigDelta:
    """
    A minimal "what-if" simulator for latency.
    You can extend this with backend-specific knobs later.
    """
    beam_width_multiplier: float = 1.0
    chunk_length_multiplier: float = 1.0
    quantization_speedup: float = 1.0  # >1 means faster


def simulate_latency_p95(base_p95_s: float, delta: ConfigDelta) -> float:
    """
    Rule-of-thumb estimator:
    - Beam width increases decoder/search cost (roughly linear-ish for small beams).
    - Chunk length changes number of calls/overheads (often inverse-ish).
    - Quantization acts as speedup multiplier.
    """
    est = base_p95_s
    est *= max(0.5, delta.beam_width_multiplier)
    est *= max(0.5, 1.0 / max(0.25, delta.chunk_length_multiplier))
    est /= max(1.0, delta.quantization_speedup)
    return float(est)
