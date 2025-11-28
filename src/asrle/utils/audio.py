from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioInfo:
    duration_s: float | None
    sample_rate: int | None


def probe_audio(path: str) -> AudioInfo:
    """
    Best-effort probe.
    - requires file existence
    - returns unknown duration if soundfile can't parse
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        import soundfile as sf

        info = sf.info(path)
        duration = float(info.frames) / float(info.samplerate) if info.frames else None
        return AudioInfo(duration_s=duration, sample_rate=int(info.samplerate))
    except Exception:
        return AudioInfo(duration_s=None, sample_rate=None)


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    # lightweight linear resample fallback (not audiophile-perfect, but robust)
    ratio = target_sr / sr
    n = int(round(len(audio) * ratio))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    out = np.interp(x_new, x_old, audio).astype(np.float32)
    return out, target_sr


def load_audio_mono_16k(path: str) -> np.ndarray:
    audio, sr = load_audio_mono(path)
    audio, _ = resample(audio, sr, 16000)
    return audio


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sr)


def slice_to_wav(
    input_path: str,
    start_s: float,
    end_s: float,
    out_path: str,
    target_sr: int = 16000,
) -> tuple[float, float]:
    """
    Writes [start_s, end_s) slice to out_path as WAV at target_sr.
    Returns (actual_start_s, actual_end_s) after clamping to audio duration (if known).
    """
    audio, sr = load_audio_mono(input_path)
    duration = len(audio) / max(1, sr)

    s = max(0.0, float(start_s))
    e = min(float(end_s), duration) if duration > 0 else float(end_s)
    if e <= s:
        # write empty-ish chunk
        write_wav(out_path, np.zeros((1,), dtype=np.float32), target_sr)
        return s, s

    i0 = int(round(s * sr))
    i1 = int(round(e * sr))
    chunk = audio[i0:i1]
    chunk, sr2 = resample(chunk, sr, target_sr)
    write_wav(out_path, chunk, sr2)
    return s, e
