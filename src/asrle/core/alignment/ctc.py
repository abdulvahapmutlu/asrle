from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asrle.types import WordStamp
from asrle.utils.audio import load_audio_mono_16k
from asrle.utils.text import normalize_text


@dataclass
class CTCAlignerConfig:
    model_name: str = "facebook/wav2vec2-base-960h"
    device: str = "cpu"  # "cpu" or "cuda"
    blank_token_id: int | None = None
    # If True, uses torchaudio forced_align when available; otherwise greedy segmentation fallback.
    prefer_forced_align: bool = True


class CTCWordAligner:
    """
    True word-level alignment using a CTC acoustic model (e.g., wav2vec2 CTC).
    It aligns *the provided reference text* to the audio.

    Fix: normalize reference text (incl. BOM removal + punctuation cleanup) before tokenization
    to prevent <unk>-heavy token streams.
    """

    def __init__(self, cfg: CTCAlignerConfig):
        self.cfg = cfg
        self._model = None
        self._proc = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._proc is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except Exception as e:
            raise RuntimeError("CTC alignment requires extras: pip install -e '.[alignment]'") from e

        self._proc = AutoProcessor.from_pretrained(self.cfg.model_name)
        self._model = AutoModelForCTC.from_pretrained(self.cfg.model_name)
        self._model.eval()
        if self.cfg.device == "cuda":
            self._model.to("cuda")

    def align_words(self, audio_path: str, reference_text: str) -> list[WordStamp]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._proc is not None

        import torch

        audio = load_audio_mono_16k(audio_path)
        if audio.size < 1600:
            return []

        inputs = self._proc(audio, sampling_rate=16000, return_tensors="pt")
        if self.cfg.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[0]  # (T, V)

        log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
        T = log_probs.shape[0]
        duration_s = float(len(audio) / 16000.0)
        sec_per_t = duration_s / max(1, T)

        # Normalize reference text (key fix for <unk> storms)
        ref_norm = normalize_text(reference_text)
        if not ref_norm:
            return []

        enc = self._proc.tokenizer(
            ref_norm,
            return_tensors="pt",
            add_special_tokens=False,
        )
        target_ids = enc.input_ids[0].tolist()

        # Determine blank token id if not specified
        blank = self.cfg.blank_token_id
        if blank is None:
            blank = int(getattr(self._proc.tokenizer, "pad_token_id", 0) or 0)

        alignment_path: list[int] | None = None
        if self.cfg.prefer_forced_align:
            try:
                import torchaudio

                lp = log_probs.detach().float().cpu()
                tgt = torch.tensor(target_ids, dtype=torch.long)
                aligned_tokens, _aligned_scores = torchaudio.functional.forced_align(lp, tgt, blank=blank)

                token_timeline = np.full((T,), fill_value=blank, dtype=np.int64)
                at = aligned_tokens.numpy()
                for t in range(T):
                    idx = int(at[t])
                    if 0 <= idx < len(target_ids):
                        token_timeline[t] = target_ids[idx]
                alignment_path = token_timeline.tolist()
            except Exception:
                alignment_path = None

        if alignment_path is None:
            timeline = torch.argmax(log_probs, dim=-1).detach().cpu().numpy().astype(np.int64).tolist()
            alignment_path = timeline

        delimiter = None
        try:
            delimiter = self._proc.tokenizer.convert_tokens_to_ids("|")
        except Exception:
            delimiter = None

        ref_decoded = self._proc.tokenizer.decode(target_ids).replace("<pad>", "").strip()
        ref_words = [w for w in ref_decoded.replace("|", " ").split() if w]

        if delimiter is None or delimiter == -1:
            n = len(ref_words)
            if n == 0:
                return []
            step = duration_s / n
            return [
                WordStamp(word=ref_words[i], start_s=i * step, end_s=(i + 1) * step, source="ctc-aligner")
                for i in range(n)
            ]

        delim_times = [i for i, tid in enumerate(alignment_path) if int(tid) == int(delimiter)]
        if not delim_times:
            n = len(ref_words)
            if n == 0:
                return []
            step = duration_s / n
            return [
                WordStamp(word=ref_words[i], start_s=i * step, end_s=(i + 1) * step, source="ctc-aligner")
                for i in range(n)
            ]

        boundaries = [0] + delim_times + [T - 1]
        word_spans: list[tuple[float, float]] = []
        for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
            s = float(b0) * sec_per_t
            e = float(b1) * sec_per_t
            if e <= s:
                e = s + 0.05
            word_spans.append((s, e))

        k = min(len(ref_words), len(word_spans))
        return [
            WordStamp(word=ref_words[i], start_s=word_spans[i][0], end_s=word_spans[i][1], source="ctc-aligner")
            for i in range(k)
        ]
