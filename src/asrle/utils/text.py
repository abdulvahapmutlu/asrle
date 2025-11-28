from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)


def normalize_text(s: str) -> str:
    # Fix: strip UTF-8 BOM (common in Windows text files) so both jiwer and CTC tokenization behave.
    s = (s or "").replace("\ufeff", "")
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def split_words(s: str) -> list[str]:
    s = normalize_text(s)
    return s.split() if s else []
